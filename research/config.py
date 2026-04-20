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

"""Configuration dataclasses for LeaRS research modules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatentConfig:
    source: str = "actor_logprob"
    layer_index: int = -1
    include_prompt: bool = False
    dtype: str = "fp16"
    capture_every_n_steps: int = 1

    def post_init(self):
        if self.source not in {"actor_logprob"}:
            raise ValueError(f"Unsupported latent source: {self.source}")
        if self.dtype not in {"fp16", "fp32"}:
            raise ValueError(f"Unsupported latent dtype: {self.dtype}")
        if self.include_prompt:
            raise ValueError(
                "research.latent.include_prompt=true is currently unsupported. "
                "Please set research.latent.include_prompt=false."
            )
        if self.capture_every_n_steps <= 0:
            raise ValueError("latent.capture_every_n_steps must be > 0")


@dataclass
class BufferConfig:
    backend: str = "parquet"
    dir: Optional[str] = None
    shard_max_samples: int = 1024
    compression: str = "zstd"
    max_disk_gb: float = 200.0
    max_train_samples: int = 100000
    flush_every_n_steps: int = 1

    def post_init(self):
        if self.backend not in {"parquet"}:
            raise ValueError(f"Unsupported buffer backend: {self.backend}")
        if self.compression not in {"zstd", "snappy", "none"}:
            raise ValueError(f"Unsupported buffer compression: {self.compression}")
        if self.dir is not None:
            self.dir = os.path.abspath(self.dir)
        if self.shard_max_samples <= 0:
            raise ValueError("buffer.shard_max_samples must be > 0")
        if self.max_disk_gb <= 0:
            raise ValueError("buffer.max_disk_gb must be > 0")
        if self.max_train_samples <= 0:
            raise ValueError("buffer.max_train_samples must be > 0")
        if self.flush_every_n_steps <= 0:
            raise ValueError("buffer.flush_every_n_steps must be > 0")


@dataclass
class ARConfig:
    model_type: str = "tiny_transformer"
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    lr: float = 1e-4
    batch_size: int = 16
    train_steps: int = 200
    train_every_n_steps: int = 50
    start_after_steps: int = 0
    min_buffer_samples: int = 256
    device: str = "cpu"
    max_seq_len: int = 2048
    num_workers: int = 0
    window_intervals: int = 0
    window_interval_steps: Optional[int] = None
    max_age_steps: Optional[int] = None
    stale_action: str = "warn"
    async_enabled: bool = False
    async_queue_size: int = 1
    reload_every_n_steps: int = 1
    continue_from_latest: bool = False

    def post_init(self):
        if self.model_type not in {"tiny_transformer"}:
            raise ValueError(f"Unsupported ar.model_type: {self.model_type}")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported ar.device: {self.device}")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("ar.{d_model,n_layers,n_heads} must be > 0")
        if self.batch_size <= 0 or self.train_steps <= 0:
            raise ValueError("ar.{batch_size,train_steps} must be > 0")
        if self.train_every_n_steps <= 0:
            raise ValueError("ar.train_every_n_steps must be > 0")
        if self.start_after_steps < 0:
            raise ValueError("ar.start_after_steps must be >= 0")
        if self.min_buffer_samples <= 0:
            raise ValueError("ar.min_buffer_samples must be > 0")
        if self.max_seq_len <= 1:
            raise ValueError("ar.max_seq_len must be > 1")
        if self.window_intervals < 0:
            raise ValueError("ar.window_intervals must be >= 0")
        if self.window_interval_steps is not None and self.window_interval_steps <= 0:
            raise ValueError("ar.window_interval_steps must be > 0 when set")
        if self.max_age_steps is not None and self.max_age_steps <= 0:
            raise ValueError("ar.max_age_steps must be > 0 when set")
        if self.stale_action not in {"warn", "fail"}:
            raise ValueError("ar.stale_action must be one of {'warn', 'fail'}")
        if self.async_queue_size <= 0:
            raise ValueError("ar.async_queue_size must be > 0")
        if self.reload_every_n_steps <= 0:
            raise ValueError("ar.reload_every_n_steps must be > 0")


@dataclass
class IntrinsicConfig:
    mode: str = "weighted_additive"
    source: str = "ar_error"
    transform: str = "identity"
    outcome_gated: bool = True  # legacy knob, kept for backward compatibility
    success_threshold: float = 0.5
    lambda_success: float = 1e-5  # legacy knob, kept for backward compatibility
    lambda_failure: float = 1e-3  # legacy knob, kept for backward compatibility
    normalize: str = "per_sequence_zscore"  # legacy alias for normalize_scope
    normalize_scope: str = "per_sequence_zscore"
    clip_value: float = 5.0
    token_mode: str = "dense"
    epsilon: float = 1e-6
    eta: float = 0.05
    gate_mode: str = "failure_only"
    lambda_success_gate: float = 0.25
    temporal_smoothing_window: int = 3
    group_norm_per_timestep: bool = False
    group_norm_scope: str = "per_timestep"

    def post_init(self):
        if self.mode not in {"weighted_additive"}:
            raise ValueError(f"Unsupported intrinsic.mode: {self.mode}")
        if self.source not in {"ar_error", "sampled_entropy", "response_length"}:
            raise ValueError("intrinsic.source must be one of {'ar_error', 'sampled_entropy', 'response_length'}")
        if self.transform not in {"identity", "sigmoid", "tanh"}:
            raise ValueError("intrinsic.transform must be one of {'identity', 'sigmoid', 'tanh'}")
        if self.normalize not in {"running_zscore", "none", "per_sequence_zscore"}:
            raise ValueError(f"Unsupported intrinsic.normalize: {self.normalize}")
        if self.normalize_scope not in {"running_zscore", "none", "per_sequence_zscore"}:
            raise ValueError(f"Unsupported intrinsic.normalize_scope: {self.normalize_scope}")
        # Backward compatibility: legacy `normalize` can still drive scope when scope is untouched/default.
        if self.normalize_scope == "per_sequence_zscore" and self.normalize in {"running_zscore", "none"}:
            self.normalize_scope = self.normalize
        self.normalize = self.normalize_scope
        if self.token_mode not in {"dense", "last_token"}:
            raise ValueError(f"Unsupported intrinsic.token_mode: {self.token_mode}")
        if self.clip_value <= 0:
            raise ValueError("intrinsic.clip_value must be > 0")
        if self.lambda_success < 0 or self.lambda_failure < 0:
            raise ValueError("intrinsic lambdas must be >= 0")
        if self.eta < 0:
            raise ValueError("intrinsic.eta must be >= 0")
        if self.gate_mode not in {"failure_only", "asymmetric", "none"}:
            raise ValueError("intrinsic.gate_mode must be one of {'failure_only', 'asymmetric', 'none'}")
        if self.lambda_success_gate < 0:
            raise ValueError("intrinsic.lambda_success_gate must be >= 0")
        if self.temporal_smoothing_window <= 0:
            raise ValueError("intrinsic.temporal_smoothing_window must be > 0")
        if self.group_norm_scope not in {"per_timestep", "pooled_sequence_group", "pooled_batch"}:
            raise ValueError(
                "intrinsic.group_norm_scope must be one of "
                "{'per_timestep', 'pooled_sequence_group', 'pooled_batch'}"
            )


@dataclass
class TracingConfig:
    enabled: bool = True
    schema_version: str = "v1"
    dir: Optional[str] = None
    flush_every_n_steps: int = 10
    compression: str = "zstd"
    save_tokens: bool = True
    save_decoded_text: bool = True
    save_latents: bool = True
    save_ar_error: bool = True
    retention_mode: str = "keep_all"
    max_disk_gb: float = 500.0
    sample_rate: float = 1.0

    def post_init(self):
        if self.dir is not None:
            self.dir = os.path.abspath(self.dir)
        if self.flush_every_n_steps <= 0:
            raise ValueError("tracing.flush_every_n_steps must be > 0")
        if self.compression not in {"zstd", "snappy", "none"}:
            raise ValueError("Unsupported tracing.compression")
        if self.retention_mode not in {"keep_all", "rolling_budget", "sampled"}:
            raise ValueError("tracing.retention_mode must be one of {'keep_all', 'rolling_budget', 'sampled'}")
        if self.max_disk_gb <= 0:
            raise ValueError("tracing.max_disk_gb must be > 0")
        if not (0.0 < self.sample_rate <= 1.0):
            raise ValueError("tracing.sample_rate must be in (0, 1]")
        if not self.schema_version.strip():
            raise ValueError("tracing.schema_version must be non-empty")


@dataclass
class ResearchConfig:
    enabled: bool = False
    latent: LatentConfig = field(default_factory=LatentConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    ar: ARConfig = field(default_factory=ARConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    checkpoint_subdir: str = "research"

    def post_init(self):
        self.checkpoint_subdir = self.checkpoint_subdir.strip() or "research"
        self.latent.post_init()
        self.buffer.post_init()
        self.ar.post_init()
        self.intrinsic.post_init()
        self.tracing.post_init()
