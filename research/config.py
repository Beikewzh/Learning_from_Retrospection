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
    min_buffer_samples: int = 256
    device: str = "cpu"
    max_seq_len: int = 2048
    num_workers: int = 0

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
        if self.min_buffer_samples <= 0:
            raise ValueError("ar.min_buffer_samples must be > 0")
        if self.max_seq_len <= 1:
            raise ValueError("ar.max_seq_len must be > 1")


@dataclass
class IntrinsicConfig:
    mode: str = "weighted_additive"
    outcome_gated: bool = True
    success_threshold: float = 0.0
    lambda_success: float = 0.1
    lambda_failure: float = 0.1
    normalize: str = "running_zscore"
    clip_value: float = 5.0
    token_mode: str = "dense"
    epsilon: float = 1e-6

    def post_init(self):
        if self.mode not in {"weighted_additive"}:
            raise ValueError(f"Unsupported intrinsic.mode: {self.mode}")
        if self.normalize not in {"running_zscore", "none"}:
            raise ValueError(f"Unsupported intrinsic.normalize: {self.normalize}")
        if self.token_mode not in {"dense", "last_token"}:
            raise ValueError(f"Unsupported intrinsic.token_mode: {self.token_mode}")
        if self.clip_value <= 0:
            raise ValueError("intrinsic.clip_value must be > 0")
        if self.lambda_success < 0 or self.lambda_failure < 0:
            raise ValueError("intrinsic lambdas must be >= 0")


@dataclass
class ResearchConfig:
    enabled: bool = False
    latent: LatentConfig = field(default_factory=LatentConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    ar: ARConfig = field(default_factory=ARConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    checkpoint_subdir: str = "research"

    def post_init(self):
        self.checkpoint_subdir = self.checkpoint_subdir.strip() or "research"
        self.latent.post_init()
        self.buffer.post_init()
        self.ar.post_init()
        self.intrinsic.post_init()
