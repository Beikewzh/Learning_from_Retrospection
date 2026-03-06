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

"""Outcome-gated weighted-additive intrinsic reward rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from research.config import IntrinsicConfig


@dataclass
class RunningZScore:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        for v in flat.tolist():
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return max(self.var, 1e-12) ** 0.5


def _normalize(ssl_error: torch.Tensor, cfg: IntrinsicConfig, stats: RunningZScore) -> torch.Tensor:
    if cfg.normalize == "none":
        return ssl_error
    return (ssl_error - stats.mean) / (stats.std + cfg.epsilon)


def apply_intrinsic_rule(
    *,
    ssl_error: torch.Tensor,
    response_mask: torch.Tensor,
    extrinsic_scores: torch.Tensor,
    cfg: IntrinsicConfig,
    stats: RunningZScore,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transform per-token SSL error into intrinsic reward tensor [B, T]."""
    mask = response_mask.float()
    normalized = _normalize(ssl_error, cfg=cfg, stats=stats)

    # Sequence-level outcome for gating uses summed extrinsic over response tokens.
    extrinsic_final = (extrinsic_scores * mask).sum(dim=-1)
    success = extrinsic_final > cfg.success_threshold

    if cfg.outcome_gated:
        signed_scale = torch.where(
            success,
            torch.full_like(extrinsic_final, -cfg.lambda_success),
            torch.full_like(extrinsic_final, +cfg.lambda_failure),
        )
    else:
        signed_scale = torch.full_like(extrinsic_final, cfg.lambda_failure)

    if cfg.token_mode == "dense":
        intrinsic = normalized * signed_scale.unsqueeze(-1)
    elif cfg.token_mode == "last_token":
        seq_signal = (normalized * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        intrinsic = torch.zeros_like(normalized)
        lengths = mask.sum(dim=-1).long().clamp_min(1)
        last_idx = lengths - 1
        intrinsic[torch.arange(intrinsic.size(0)), last_idx] = seq_signal * signed_scale
    else:
        raise ValueError(f"Unsupported intrinsic token_mode: {cfg.token_mode}")

    intrinsic = torch.clamp(intrinsic, min=-cfg.clip_value, max=cfg.clip_value)
    intrinsic = intrinsic * mask

    metrics = {
        "research/intrinsic/success_rate": success.float().mean().item(),
        "research/intrinsic/mean": intrinsic.mean().item(),
        "research/intrinsic/max": intrinsic.max().item(),
        "research/intrinsic/min": intrinsic.min().item(),
        "research/intrinsic/z_mean": normalized.mean().item(),
        "research/intrinsic/z_std": normalized.std(unbiased=False).item(),
    }
    return intrinsic, metrics
