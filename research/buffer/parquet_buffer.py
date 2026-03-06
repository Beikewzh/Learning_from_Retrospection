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

"""Arrow/Parquet-backed append-only latent buffer."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from research.latent.codec import deserialize_latent_tensor, serialize_latent_tensor


@dataclass
class BufferStats:
    total_samples: int = 0
    total_shards: int = 0


class ParquetLatentBuffer:
    """Append-only latent buffer with shard rotation and disk-budget retention."""

    def __init__(
        self,
        root_dir: str,
        shard_max_samples: int,
        compression: str,
        max_disk_gb: float,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.shard_dir = os.path.join(self.root_dir, "shards")
        self.manifest_path = os.path.join(self.root_dir, "manifest.json")
        self.shard_max_samples = int(shard_max_samples)
        self.compression = None if compression == "none" else compression
        self.max_disk_bytes = int(max_disk_gb * (1024**3))

        os.makedirs(self.shard_dir, exist_ok=True)
        self._pending_rows: list[dict[str, Any]] = []
        self._next_shard_idx = self._discover_next_shard_idx()
        self.stats = self._load_stats()

    def _discover_next_shard_idx(self) -> int:
        indices = []
        for fname in os.listdir(self.shard_dir):
            if not fname.startswith("part-") or not fname.endswith(".parquet"):
                continue
            try:
                indices.append(int(fname.split("-")[1].split(".")[0]))
            except Exception:
                continue
        return (max(indices) + 1) if indices else 0

    def _load_stats(self) -> BufferStats:
        if not os.path.exists(self.manifest_path):
            return BufferStats(total_samples=0, total_shards=0)
        with open(self.manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return BufferStats(total_samples=int(data.get("total_samples", 0)), total_shards=int(data.get("total_shards", 0)))

    def _save_stats(self) -> None:
        payload = {
            "total_samples": self.stats.total_samples,
            "total_shards": self.stats.total_shards,
            "next_shard_idx": self._next_shard_idx,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _rows_to_table(self, rows: list[dict[str, Any]]) -> pa.Table:
        return pa.table(
            {
                "uid": pa.array([r["uid"] for r in rows], type=pa.string()),
                "step": pa.array([r["step"] for r in rows], type=pa.int64()),
                "response_length": pa.array([r["response_length"] for r in rows], type=pa.int32()),
                "hidden_dim": pa.array([r["hidden_dim"] for r in rows], type=pa.int32()),
                "dtype": pa.array([r["dtype"] for r in rows], type=pa.string()),
                "extrinsic_final": pa.array([r["extrinsic_final"] for r in rows], type=pa.float32()),
                "success": pa.array([r["success"] for r in rows], type=pa.bool_()),
                "latent_blob": pa.array([r["latent_blob"] for r in rows], type=pa.binary()),
            }
        )

    def _write_shard(self, rows: list[dict[str, Any]]) -> str:
        fname = f"part-{self._next_shard_idx:06d}.parquet"
        self._next_shard_idx += 1
        path = os.path.join(self.shard_dir, fname)
        table = self._rows_to_table(rows)
        pq.write_table(table, path, compression=self.compression)
        self.stats.total_shards += 1
        self.stats.total_samples += len(rows)
        self._save_stats()
        return path

    def _current_disk_usage(self) -> int:
        total = 0
        for fname in os.listdir(self.shard_dir):
            path = os.path.join(self.shard_dir, fname)
            if os.path.isfile(path):
                total += os.path.getsize(path)
        return total

    def _enforce_disk_budget(self) -> None:
        if self._current_disk_usage() <= self.max_disk_bytes:
            return
        shard_files = sorted(
            [f for f in os.listdir(self.shard_dir) if f.startswith("part-") and f.endswith(".parquet")]
        )
        for fname in shard_files:
            if self._current_disk_usage() <= self.max_disk_bytes:
                break
            path = os.path.join(self.shard_dir, fname)
            try:
                table = pq.read_table(path, columns=["uid"])
                removed_samples = table.num_rows
            except Exception:
                removed_samples = 0
            os.remove(path)
            self.stats.total_shards = max(0, self.stats.total_shards - 1)
            self.stats.total_samples = max(0, self.stats.total_samples - removed_samples)
        self._save_stats()

    def append_batch(
        self,
        *,
        uids: Iterable[str],
        step: int,
        latents: torch.Tensor,
        response_lengths: torch.Tensor,
        extrinsic_final: torch.Tensor,
        success: torch.Tensor,
        dtype: str,
    ) -> int:
        """Append a batch of per-sample latent sequences [B, T, H]."""
        if latents.ndim != 3:
            raise ValueError(f"Expected latents [B, T, H], got {tuple(latents.shape)}")

        uids = list(uids)
        bsz = latents.size(0)
        if len(uids) != bsz:
            raise ValueError(f"uids length mismatch: {len(uids)} != {bsz}")

        appended = 0
        for i in range(bsz):
            seq_len = int(response_lengths[i].item())
            if seq_len <= 1:
                continue
            sample_latent = latents[i, :seq_len, :]
            blob, serialized_len, hidden_dim, serialized_dtype = serialize_latent_tensor(sample_latent, dtype=dtype)
            self._pending_rows.append(
                {
                    "uid": str(uids[i]),
                    "step": int(step),
                    "response_length": serialized_len,
                    "hidden_dim": hidden_dim,
                    "dtype": serialized_dtype,
                    "extrinsic_final": float(extrinsic_final[i].item()),
                    "success": bool(success[i].item()),
                    "latent_blob": blob,
                }
            )
            appended += 1

        if len(self._pending_rows) >= self.shard_max_samples:
            self.flush()
        return appended

    def flush(self) -> None:
        if not self._pending_rows:
            return
        rows = self._pending_rows
        self._pending_rows = []
        self._write_shard(rows)
        self._enforce_disk_budget()

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        shard_files = sorted(
            [f for f in os.listdir(self.shard_dir) if f.startswith("part-") and f.endswith(".parquet")]
        )
        for fname in shard_files:
            path = os.path.join(self.shard_dir, fname)
            table = pq.read_table(path)
            data = table.to_pydict()
            n = table.num_rows
            for i in range(n):
                yield {
                    "uid": data["uid"][i],
                    "step": data["step"][i],
                    "response_length": data["response_length"][i],
                    "hidden_dim": data["hidden_dim"][i],
                    "dtype": data["dtype"][i],
                    "extrinsic_final": data["extrinsic_final"][i],
                    "success": data["success"][i],
                    "latent_blob": data["latent_blob"][i],
                }

    def load_sequences(self, max_samples: int, seed: int = 0) -> list[torch.Tensor]:
        rows = list(self.iter_rows())
        if not rows:
            return []
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]

        sequences: list[torch.Tensor] = []
        for row in rows:
            seq = deserialize_latent_tensor(
                row["latent_blob"],
                seq_len=int(row["response_length"]),
                hidden_dim=int(row["hidden_dim"]),
                dtype=str(row["dtype"]),
            )
            sequences.append(seq)
        return sequences

    def get_stats(self) -> dict[str, Any]:
        return {
            "buffer/total_samples": self.stats.total_samples,
            "buffer/total_shards": self.stats.total_shards,
            "buffer/pending_rows": len(self._pending_rows),
        }
