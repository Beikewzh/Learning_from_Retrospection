# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Asynchronous trace writer for token/latent/intrinsic diagnostics."""

from __future__ import annotations

import json
import os
import queue
import random
import threading
from dataclasses import dataclass
from typing import Any

import torch

from research.config import TracingConfig


@dataclass
class TraceRecord:
    payload: dict[str, Any]
    latent: torch.Tensor | None = None


class AsyncTraceWriter:
    def __init__(self, root_dir: str, config: TracingConfig, queue_size: int = 4096):
        self.config = config
        self.root_dir = os.path.abspath(root_dir)
        self.records_dir = os.path.join(self.root_dir, "records")
        self.latents_dir = os.path.join(self.root_dir, "latents")
        os.makedirs(self.records_dir, exist_ok=True)
        os.makedirs(self.latents_dir, exist_ok=True)

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop_token = object()
        self._flush_token = object()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._shard_idx = 0
        self._records_in_shard = 0
        self._max_records_per_shard = 5000
        self._record_fh = None
        self._dropped = 0
        self._rng = random.Random(0)
        self._thread.start()

    def _open_next_shard(self) -> None:
        if self._record_fh is not None:
            self._record_fh.flush()
            self._record_fh.close()
        self._shard_idx += 1
        shard_path = os.path.join(self.records_dir, f"part-{self._shard_idx:06d}.jsonl")
        self._record_fh = open(shard_path, "a", encoding="utf-8")
        self._records_in_shard = 0

    def _safe_flush(self) -> None:
        if self._record_fh is not None:
            self._record_fh.flush()

    def _current_disk_usage(self) -> int:
        total = 0
        for root in (self.records_dir, self.latents_dir):
            for fname in os.listdir(root):
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    total += os.path.getsize(fpath)
        return total

    def _enforce_retention(self) -> None:
        if self.config.retention_mode == "keep_all":
            return
        if self.config.retention_mode == "sampled":
            return

        max_bytes = int(self.config.max_disk_gb * (1024**3))
        while self._current_disk_usage() > max_bytes:
            record_files = sorted(
                f for f in os.listdir(self.records_dir) if f.startswith("part-") and f.endswith(".jsonl")
            )
            if not record_files:
                break
            oldest = os.path.join(self.records_dir, record_files[0])
            os.remove(oldest)

            latent_files = sorted(f for f in os.listdir(self.latents_dir) if f.endswith(".pt"))
            if latent_files:
                os.remove(os.path.join(self.latents_dir, latent_files[0]))

    def enqueue(self, payload: dict[str, Any], latent: torch.Tensor | None = None) -> None:
        if not self.config.enabled:
            return
        if self.config.retention_mode == "sampled" and self._rng.random() > self.config.sample_rate:
            return
        item = TraceRecord(payload=payload, latent=latent)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    def flush(self) -> None:
        if not self.config.enabled:
            return
        try:
            self._queue.put_nowait(self._flush_token)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop_token:
                break
            if item is self._flush_token:
                self._safe_flush()
                continue
            assert isinstance(item, TraceRecord)
            if self._record_fh is None or self._records_in_shard >= self._max_records_per_shard:
                self._open_next_shard()

            payload = dict(item.payload)
            payload["schema_version"] = self.config.schema_version
            latent = item.latent
            if self.config.save_latents and latent is not None:
                uid = str(payload.get("uid", "unknown"))
                step = int(payload.get("step", -1))
                latent_name = f"latent_s{step}_u{uid}.pt"
                latent_path = os.path.join(self.latents_dir, latent_name)
                torch.save(latent.detach().to(device="cpu"), latent_path)
                payload["latent_path"] = os.path.relpath(latent_path, self.root_dir)
            else:
                payload["latent_path"] = None

            self._record_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._records_in_shard += 1
            if self._records_in_shard % max(1, self.config.flush_every_n_steps) == 0:
                self._safe_flush()
            self._enforce_retention()

        self._safe_flush()
        if self._record_fh is not None:
            self._record_fh.close()
            self._record_fh = None

    def close(self) -> None:
        if not self.config.enabled:
            return
        self._queue.put(self._stop_token)
        self._thread.join(timeout=30)

    @property
    def dropped_records(self) -> int:
        return self._dropped
