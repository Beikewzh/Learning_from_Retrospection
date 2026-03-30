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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch

from research.ar import ARScorer, ARTrainer
from research.buffer import ParquetLatentBuffer
from research.config import ResearchConfig
from research.intrinsic import RunningZScore, apply_intrinsic_rule, compute_intrinsic_gate
from research.trace_writer import AsyncTraceWriter


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
        self._recent_sequences: deque[tuple[int, torch.Tensor]] = deque(maxlen=max(1, int(config.buffer.max_train_samples)))
        self._steps_since_flush = 0
        self._ar_executor: Optional[ThreadPoolExecutor] = None
        self._ar_futures: list[Future] = []
        if self.config.ar.async_enabled:
            self._ar_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lears-ar")
        trace_dir = (
            self.config.tracing.dir
            if self.config.tracing.dir is not None
            else os.path.join(self.root_dir, "traces")
        )
        self.trace_writer: Optional[AsyncTraceWriter] = (
            AsyncTraceWriter(root_dir=trace_dir, config=self.config.tracing) if self.config.tracing.enabled else None
        )

        self.last_train_step = 0
        self.last_attempt_step = -1
        self.first_train_step = -1
        self._scorer: Optional[ARScorer] = None
        self._scorer_ckpt_mtime_ns: int | None = None
        self._last_reload_step = -1
        self._run_id = os.path.basename(self.checkpoint_root.rstrip(os.sep)) or "unknown_run"
        self._load_state()
        self._try_load_latest_scorer()

    def _try_load_latest_scorer(self) -> bool:
        latest_path = os.path.join(self.ar_dir, "checkpoints", "latest.pt")
        if not os.path.exists(latest_path):
            self._scorer = None
            self._scorer_ckpt_mtime_ns = None
            return False
        mtime_ns = int(os.stat(latest_path).st_mtime_ns)
        if self._scorer is not None and self._scorer_ckpt_mtime_ns == mtime_ns:
            return False
        self._scorer = ARScorer.load_from_checkpoint(latest_path, device=self.config.ar.device)
        self._scorer_ckpt_mtime_ns = mtime_ns
        return True

    def _maybe_reload_scorer(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        if self.config.ar.reload_every_n_steps <= 1:
            reloaded = self._try_load_latest_scorer()
            if reloaded:
                self._last_reload_step = int(global_step)
            return reloaded
        if self._last_reload_step >= 0 and (global_step - self._last_reload_step) < self.config.ar.reload_every_n_steps:
            return False
        reloaded = self._try_load_latest_scorer()
        self._last_reload_step = int(global_step)
        return reloaded

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
            "research/ar/scorer_reloaded": 0.0,
            "research/ar/async_enqueued": 0.0,
            "research/ar/async_busy": 0.0,
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

    def _poll_async_train_results(self) -> dict[str, float]:
        metrics = {
            "research/ar/async_completed": 0.0,
            "research/ar/async_failed": 0.0,
            "research/ar/async_queue_depth": float(len(self._ar_futures)),
        }
        if not self._ar_futures:
            return metrics

        remaining: list[Future] = []
        completed_outputs = []
        for future in self._ar_futures:
            if not future.done():
                remaining.append(future)
                continue
            try:
                output = future.result()
            except Exception as e:
                import traceback
                print(f"[LeaRS] AR async training failed: {e}\n{traceback.format_exc()}", flush=True)
                metrics["research/ar/async_failed"] += 1.0
                continue
            metrics["research/ar/async_completed"] += 1.0
            if output is not None:
                completed_outputs.append(output)

        self._ar_futures = remaining
        metrics["research/ar/async_queue_depth"] = float(len(self._ar_futures))
        if not completed_outputs:
            return metrics

        latest = max(completed_outputs, key=lambda out: int(getattr(out, "global_step", self.last_attempt_step)))
        self.last_train_step = int(getattr(latest, "global_step", self.last_attempt_step))
        if self.first_train_step < 0:
            self.first_train_step = self.last_train_step
        self._try_load_latest_scorer()
        self._save_state()
        return metrics

    def _sample_recent_sequences(
        self,
        *,
        max_samples: int,
        min_step: int | None,
        max_step: int | None,
        seed: int,
    ) -> list[torch.Tensor]:
        candidates: list[torch.Tensor] = []
        for step, seq in self._recent_sequences:
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            candidates.append(seq)
        if not candidates:
            return []
        rng = np.random.default_rng(seed)
        idx = np.arange(len(candidates))
        rng.shuffle(idx)
        chosen = idx[: max_samples]
        return [candidates[i] for i in chosen]

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

        sequences = self._sample_recent_sequences(
            max_samples=self.config.buffer.max_train_samples,
            seed=global_step,
            min_step=window_min_step,
            max_step=global_step,
        )
        if len(sequences) < self.config.ar.min_buffer_samples:
            sequences = self.buffer.load_sequences(
                max_samples=self.config.buffer.max_train_samples,
                seed=global_step,
                min_step=window_min_step,
                max_step=global_step,
            )
        metrics["research/ar/window_min_step"] = float(window_min_step if window_min_step is not None else 0)
        metrics["research/ar/windowed_samples"] = float(len(sequences))
        if len(sequences) < self.config.ar.min_buffer_samples:
            metrics["research/ar/skipped"] = 1.0
            metrics["research/ar/skip_insufficient_samples"] = 1.0
            self._save_state()
            return metrics

        if self.config.ar.async_enabled and self._ar_executor is not None:
            if len(self._ar_futures) >= self.config.ar.async_queue_size:
                metrics["research/ar/skipped"] = 1.0
                metrics["research/ar/async_busy"] = 1.0
                return metrics
            future = self._ar_executor.submit(self.ar_trainer.train_from_sequences, sequences, global_step)
            self._ar_futures.append(future)
            metrics["research/ar/async_enqueued"] = 1.0
            return metrics

        train_output = self.ar_trainer.train_from_sequences(sequences=sequences, global_step=global_step)
        if train_output is None:
            metrics["research/ar/skipped"] = 1.0
            metrics["research/ar/skip_insufficient_samples"] = 1.0
            self._save_state()
            return metrics

        self.last_train_step = int(getattr(train_output, "global_step", global_step))
        if self.first_train_step < 0:
            self.first_train_step = self.last_train_step
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
        for i in range(latents.size(0)):
            seq_len = int(response_lengths[i].item())
            if seq_len <= 1:
                continue
            self._recent_sequences.append((int(global_step), latents[i, :seq_len, :].detach().cpu().float()))

        self._steps_since_flush += 1
        if self._steps_since_flush >= self.config.buffer.flush_every_n_steps:
            self.buffer.flush()
            self._steps_since_flush = 0

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

        self._trace_step(
            batch=batch,
            response_mask=response_mask,
            latents=latents,
            ssl_error=ssl_error,
            intrinsic=intrinsic,
            extrinsic_scores=extrinsic_scores,
            global_step=global_step,
        )
        return intrinsic, metrics

    def collect_and_score(self, *, batch, extrinsic_scores: torch.Tensor, global_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.enabled:
            return torch.zeros_like(extrinsic_scores, dtype=torch.float32), {}

        metrics = {}
        metrics.update(self._poll_async_train_results())
        if self._maybe_reload_scorer(global_step=global_step):
            metrics["research/ar/scorer_reloaded"] = 1.0
        metrics.update(self._append_to_buffer(batch=batch, extrinsic_scores=extrinsic_scores, global_step=global_step))
        metrics.update(self.maybe_train_ar(global_step=global_step))
        intrinsic, intrinsic_metrics = self.compute_intrinsic(
            batch=batch,
            extrinsic_scores=extrinsic_scores,
            global_step=global_step,
        )
        metrics.update(intrinsic_metrics)
        if self.trace_writer is not None:
            metrics["research/tracing/dropped_records"] = float(self.trace_writer.dropped_records)
        self._save_state()
        return intrinsic, metrics

    def _trace_step(
        self,
        *,
        batch,
        response_mask: torch.Tensor,
        latents: torch.Tensor,
        ssl_error: torch.Tensor,
        intrinsic: torch.Tensor,
        extrinsic_scores: torch.Tensor,
        global_step: int,
    ) -> None:
        if self.trace_writer is None:
            return
        non_tensor_batch = getattr(batch, "non_tensor_batch", None)
        if non_tensor_batch is not None and "uid" in non_tensor_batch:
            uids = non_tensor_batch["uid"].tolist()
        else:
            uids = [str(i) for i in range(latents.size(0))]
        responses = batch.batch.get("responses", None)
        decoded_batch = None
        if non_tensor_batch is not None:
            for key in ("decoded_text", "response_text", "responses_text"):
                if key in non_tensor_batch:
                    decoded_batch = non_tensor_batch[key].tolist()
                    break
        response_mask_b = response_mask.bool()
        extrinsic_final = (extrinsic_scores * response_mask.float()).sum(dim=-1)
        success = (extrinsic_final > self.config.intrinsic.success_threshold).tolist()
        gate, _, _ = compute_intrinsic_gate(
            extrinsic_scores=extrinsic_scores,
            response_mask=response_mask,
            cfg=self.config.intrinsic,
        )

        for i, uid in enumerate(uids):
            mask = response_mask_b[i]
            seq_len = int(mask.sum().item())
            ar_vals = ssl_error[i, :seq_len].detach().cpu()
            intrinsic_vals = intrinsic[i, :seq_len].detach().cpu()
            gate_i = float(gate[i].item()) if gate.ndim > 0 else float(gate.item())
            intrinsic_contrib = (self.config.intrinsic.eta * gate_i * intrinsic_vals).detach().cpu()
            payload = {
                "run_id": self._run_id,
                "step": int(global_step),
                "sample_idx": int(i),
                "uid": str(uid),
                "response_length": int(seq_len),
                "success": bool(success[i]),
                "extrinsic_final": float(extrinsic_final[i].item()),
                "intrinsic_eta": float(self.config.intrinsic.eta),
                "intrinsic_gate": float(gate_i),
                "ar_error_mean": float(ar_vals.mean().item()) if seq_len > 0 else 0.0,
                "ar_error_std": float(ar_vals.std(unbiased=False).item()) if seq_len > 1 else 0.0,
                "intrinsic_mean": float(intrinsic_vals.mean().item()) if seq_len > 0 else 0.0,
                "intrinsic_std": float(intrinsic_vals.std(unbiased=False).item()) if seq_len > 1 else 0.0,
                "zscore_mean": float(intrinsic_vals.mean().item()) if seq_len > 0 else 0.0,
                "zscore_std": float(intrinsic_vals.std(unbiased=False).item()) if seq_len > 1 else 0.0,
                "intrinsic_contrib_mean": float(intrinsic_contrib.mean().item()) if seq_len > 0 else 0.0,
                "intrinsic_contrib_std": float(intrinsic_contrib.std(unbiased=False).item()) if seq_len > 1 else 0.0,
                "ar_error_tokens": ar_vals.tolist() if self.config.tracing.save_ar_error else None,
                "intrinsic_tokens": intrinsic_vals.tolist(),
                "intrinsic_contrib_tokens": intrinsic_contrib.tolist(),
                "decoded_text": (
                    decoded_batch[i]
                    if (self.config.tracing.save_decoded_text and decoded_batch is not None and i < len(decoded_batch))
                    else None
                ),
            }
            if self.config.tracing.save_tokens and responses is not None:
                payload["response_token_ids"] = responses[i, :seq_len].detach().cpu().long().tolist()
            else:
                payload["response_token_ids"] = None
            latent = latents[i, :seq_len, :].detach().cpu() if self.config.tracing.save_latents else None
            self.trace_writer.enqueue(payload=payload, latent=latent)

    def close(self) -> None:
        self.buffer.flush()
        if self._ar_futures:
            for future in self._ar_futures:
                try:
                    future.result(timeout=120)
                except Exception:
                    pass
            self._poll_async_train_results()
        if self._ar_executor is not None:
            self._ar_executor.shutdown(wait=True)
            self._ar_executor = None
        if self.trace_writer is not None:
            self.trace_writer.flush()
            self.trace_writer.close()
            self.trace_writer = None
        self._save_state()
