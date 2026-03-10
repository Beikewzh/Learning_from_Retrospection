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

"""Coordinator for latent buffering, AR training cadence, and intrinsic scoring."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch

from research.ar import ARScorer, ARTrainer
from research.buffer import ParquetLatentBuffer
from research.config import ResearchConfig
from research.intrinsic import RunningZScore, apply_intrinsic_rule


class LeaRSManager:
    def __init__(self, config: ResearchConfig, checkpoint_root: str):
        self.config = config
        self.enabled = bool(config.enabled)
        self.checkpoint_root = os.path.abspath(checkpoint_root)

        self.root_dir = (
            config.buffer.dir
            if config.buffer.dir is not None
            else os.path.join(self.checkpoint_root, config.checkpoint_subdir)
        )
        self.buffer_dir = os.path.join(self.root_dir, "buffer")
        self.ar_dir = os.path.join(self.root_dir, "ar")
        self.state_path = os.path.join(self.root_dir, "state.json")
        os.makedirs(self.buffer_dir, exist_ok=True)
        os.makedirs(self.ar_dir, exist_ok=True)

        self.buffer = ParquetLatentBuffer(
            root_dir=self.buffer_dir,
            shard_max_samples=config.buffer.shard_max_samples,
            compression=config.buffer.compression,
            max_disk_gb=config.buffer.max_disk_gb,
        )
        self.ar_trainer = ARTrainer(config=config.ar, work_dir=self.ar_dir)
        self.running_stats = RunningZScore()

        self.last_train_step = 0
        self.last_attempt_step = -1
        self.first_train_step = -1
        self._scorer: Optional[ARScorer] = None
        self._load_state()
        self._try_load_latest_scorer()

    def _try_load_latest_scorer(self) -> bool:
        latest_path = os.path.join(self.ar_dir, "checkpoints", "latest.pt")
        if not os.path.exists(latest_path):
            self._scorer = None
            return False
        self._scorer = ARScorer.load_from_checkpoint(latest_path, device=self.config.ar.device)
        return True

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path, encoding="utf-8") as f:
            state = json.load(f)
        self.last_train_step = int(state.get("last_train_step", 0))
        self.last_attempt_step = int(state.get("last_attempt_step", -1))
        self.first_train_step = int(state.get("first_train_step", -1))
        stats = state.get("running_stats", {})
        self.running_stats.count = int(stats.get("count", 0))
        self.running_stats.mean = float(stats.get("mean", 0.0))
        self.running_stats.m2 = float(stats.get("m2", 0.0))

    def _save_state(self) -> None:
        state = {
            "last_train_step": self.last_train_step,
            "last_attempt_step": self.last_attempt_step,
            "first_train_step": self.first_train_step,
            "running_stats": {
                "count": self.running_stats.count,
                "mean": self.running_stats.mean,
                "m2": self.running_stats.m2,
            },
            "config": asdict(self.config),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def should_capture(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        return (global_step % self.config.latent.capture_every_n_steps) == 0

    def build_capture_meta(self, global_step: int) -> dict[str, Any]:
        capture = self.should_capture(global_step)
        return {
            "capture_latent": capture,
            "latent_layer_index": self.config.latent.layer_index,
            "latent_include_prompt": self.config.latent.include_prompt,
            "latent_dtype": self.config.latent.dtype,
        }

    def _base_ar_metrics(self) -> dict[str, float]:
        return {
            "research/ar/attempted": 0.0,
            "research/ar/trained": 0.0,
            "research/ar/skipped": 0.0,
            "research/ar/skip_cadence": 0.0,
            "research/ar/skip_warmup": 0.0,
            "research/ar/skip_insufficient_samples": 0.0,
            "research/ar/window_intervals": float(self.config.ar.window_intervals),
            "research/ar/window_min_step": 0.0,
            "research/ar/windowed_samples": 0.0,
            "research/ar/last_attempt_step": float(self.last_attempt_step),
            "research/ar/last_train_step": float(self.last_train_step),
            "research/ar/first_train_step": float(self.first_train_step),
            "research/ar/start_after_steps": float(self.config.ar.start_after_steps),
            "research/ar/train_every_n_steps": float(self.config.ar.train_every_n_steps),
        }

    def maybe_train_ar(self, global_step: int) -> dict[str, Any]:
        metrics = self._base_ar_metrics()
        if not self.enabled:
            return metrics

        if global_step < self.config.ar.start_after_steps:
            metrics["research/ar/skipped"] = 1.0
            metrics["research/ar/skip_warmup"] = 1.0
            return metrics

        # Attempt on deterministic cadence: start_after_steps, start_after_steps + k * train_every_n_steps.
        if (global_step - self.config.ar.start_after_steps) % self.config.ar.train_every_n_steps != 0:
            metrics["research/ar/skipped"] = 1.0
            metrics["research/ar/skip_cadence"] = 1.0
            return metrics

        # Avoid duplicate attempt bookkeeping if called repeatedly at the same step.
        if global_step == self.last_attempt_step:
            metrics["research/ar/skipped"] = 1.0
            metrics["research/ar/skip_cadence"] = 1.0
            return metrics

        self.last_attempt_step = global_step
        metrics["research/ar/attempted"] = 1.0
        metrics["research/ar/last_attempt_step"] = float(self.last_attempt_step)

        window_min_step: int | None = None
        if self.config.ar.window_intervals > 0:
            interval_steps = (
                self.config.ar.window_interval_steps
                if self.config.ar.window_interval_steps is not None
                else self.config.ar.train_every_n_steps
            )
            window_span = interval_steps * self.config.ar.window_intervals
            window_min_step = max(0, int(global_step - window_span + 1))

        sequences = self.buffer.load_sequences(
            max_samples=self.config.buffer.max_train_samples,
            seed=global_step,
            min_step=window_min_step,
            max_step=global_step,
        )
        metrics["research/ar/window_min_step"] = float(window_min_step if window_min_step is not None else 0)
        metrics["research/ar/windowed_samples"] = float(len(sequences))
        train_output = self.ar_trainer.train_from_sequences(sequences=sequences, global_step=global_step)
        if train_output is None:
            metrics["research/ar/skipped"] = 1.0
            metrics["research/ar/skip_insufficient_samples"] = 1.0
            self._save_state()
            return metrics

        self.last_train_step = global_step
        if self.first_train_step < 0:
            self.first_train_step = global_step
        metrics["research/ar/trained"] = 1.0
        metrics["research/ar/train_loss"] = float(train_output.train_loss)
        metrics["research/ar/train_steps"] = float(train_output.steps)
        metrics["research/ar/last_train_step"] = float(self.last_train_step)
        metrics["research/ar/first_train_step"] = float(self.first_train_step)
        self._save_state()
        self._try_load_latest_scorer()
        return metrics

    def _append_to_buffer(self, batch, extrinsic_scores: torch.Tensor, global_step: int) -> dict[str, Any]:
        if "latent_response_last" not in batch.batch:
            return {"research/latent/captured": 0.0}

        latents: torch.Tensor = batch.batch["latent_response_last"]
        response_mask: torch.Tensor = batch.batch["response_mask"]
        response_lengths = response_mask.sum(dim=-1).long()
        if "uid" in batch.non_tensor_batch:
            uids = batch.non_tensor_batch["uid"].tolist()
        else:
            uids = [str(i) for i in range(latents.size(0))]

        extrinsic_final = (extrinsic_scores * response_mask).sum(dim=-1)
        success = extrinsic_final > self.config.intrinsic.success_threshold
        self.buffer.append_batch(
            uids=uids,
            step=global_step,
            latents=latents,
            response_lengths=response_lengths,
            extrinsic_final=extrinsic_final,
            success=success,
            dtype=self.config.latent.dtype,
        )
        self.buffer.flush()

        metrics = {
            "research/latent/captured": float(latents.size(0)),
            "research/latent/hidden_dim": float(latents.size(-1)),
        }
        metrics.update({f"research/{k}": v for k, v in self.buffer.get_stats().items()})
        return metrics

    def compute_intrinsic(
        self,
        batch,
        extrinsic_scores: torch.Tensor,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        response_mask = batch.batch["response_mask"]
        zeros = torch.zeros_like(response_mask, dtype=torch.float32)

        if (self._scorer is None) or ("latent_response_last" not in batch.batch):
            return zeros, {
                "research/intrinsic/active": 0.0,
                "research/ar/global_step": -1.0,
                "research/ar/train_loss": 0.0,
                "research/ar/age_steps": 0.0,
                "research/ar/stale": 0.0,
            }

        latents = batch.batch["latent_response_last"]
        ssl_error = self._scorer.score(latents=latents, response_mask=response_mask)
        ssl_valid_mask = response_mask.bool().clone()
        if ssl_valid_mask.size(1) > 0:
            ssl_valid_mask[:, 0] = False
        masked = ssl_error[ssl_valid_mask]
        self.running_stats.update(masked)

        intrinsic, metrics = apply_intrinsic_rule(
            ssl_error=ssl_error,
            response_mask=response_mask,
            intrinsic_mask=ssl_valid_mask,
            extrinsic_scores=extrinsic_scores,
            cfg=self.config.intrinsic,
            stats=self.running_stats,
        )
        metrics["research/intrinsic/active"] = 1.0
        metrics["research/ar/global_step"] = float(self._scorer.meta.global_step)
        metrics["research/ar/train_loss"] = float(self._scorer.meta.train_loss)
        age_steps = max(0, int(global_step) - int(self._scorer.meta.global_step))
        metrics["research/ar/age_steps"] = float(age_steps)
        metrics["research/ar/stale"] = 0.0
        if self.config.ar.max_age_steps is not None and age_steps > self.config.ar.max_age_steps:
            stale_msg = (
                f"AR scorer is stale by {age_steps} steps at global_step={global_step} "
                f"(max_age_steps={self.config.ar.max_age_steps})."
            )
            metrics["research/ar/stale"] = 1.0
            if self.config.ar.stale_action == "fail":
                raise RuntimeError(stale_msg)
            warnings.warn(stale_msg)
        return intrinsic.cpu(), metrics

    def collect_and_score(self, *, batch, extrinsic_scores: torch.Tensor, global_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.enabled:
            return torch.zeros_like(extrinsic_scores, dtype=torch.float32), {}

        metrics = {}
        metrics.update(self._append_to_buffer(batch=batch, extrinsic_scores=extrinsic_scores, global_step=global_step))
        metrics.update(self.maybe_train_ar(global_step=global_step))
        intrinsic, intrinsic_metrics = self.compute_intrinsic(
            batch=batch,
            extrinsic_scores=extrinsic_scores,
            global_step=global_step,
        )
        metrics.update(intrinsic_metrics)
        self._save_state()
        return intrinsic, metrics

    def close(self) -> None:
        self.buffer.flush()
        self._save_state()
