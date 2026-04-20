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

"""Token-level intrinsic processing and gating utilities."""

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
        n = int(flat.numel())
        batch_mean = float(flat.mean().item())
        # Use population variance then map to M2 for stable merges.
        batch_var = float(flat.var(unbiased=False).item())
        batch_m2 = batch_var * n

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += batch_m2 + (delta * delta) * self.count * n / total
        self.count = total

    @property
    def var(self) -> float:
        if self.count < 2:
            return 1.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return max(self.var, 1e-12) ** 0.5


def _normalize(
    ssl_error: torch.Tensor,
    cfg: IntrinsicConfig,
    stats: RunningZScore,
    mask: torch.Tensor,
) -> torch.Tensor:
    if cfg.normalize_scope == "none":
        return ssl_error
    if cfg.normalize_scope == "running_zscore":
        return (ssl_error - stats.mean) / (stats.std + cfg.epsilon)
    if cfg.normalize_scope == "per_sequence_zscore":
        return _per_sequence_zscore(ssl_error, mask=mask, epsilon=cfg.epsilon)
    raise ValueError(f"Unsupported normalize_scope: {cfg.normalize_scope}")


def _causal_moving_average(values: torch.Tensor, mask: torch.Tensor, window: int, epsilon: float) -> torch.Tensor:
    if window <= 1:
        return values * mask
    masked_values = values * mask
    csum = torch.cumsum(masked_values, dim=-1)
    msum = torch.cumsum(mask, dim=-1)

    csum_prev = torch.zeros_like(csum)
    msum_prev = torch.zeros_like(msum)
    csum_prev[:, window:] = csum[:, :-window]
    msum_prev[:, window:] = msum[:, :-window]

    numer = csum - csum_prev
    denom = (msum - msum_prev).clamp_min(epsilon)
    return (numer / denom) * mask


def _per_sequence_zscore(values: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=-1, keepdim=True) / denom
    var = ((values - mean).pow(2) * mask).sum(dim=-1, keepdim=True) / denom
    std = (var + epsilon).sqrt()
    return (values - mean) / std


def _transform_intrinsic(values: torch.Tensor, cfg: IntrinsicConfig) -> torch.Tensor:
    if cfg.transform == "identity":
        return values
    if cfg.transform == "sigmoid":
        return torch.sigmoid(values)
    if cfg.transform == "tanh":
        return torch.tanh(values)
    raise ValueError(f"Unsupported intrinsic transform: {cfg.transform}")


def _gate_failure_only(success: torch.Tensor, cfg: IntrinsicConfig) -> torch.Tensor:
    return (~success).float()


def _gate_asymmetric(success: torch.Tensor, cfg: IntrinsicConfig) -> torch.Tensor:
    return torch.where(success, torch.full_like(success.float(), -cfg.lambda_success_gate), torch.ones_like(success.float()))


def _gate_none(success: torch.Tensor, cfg: IntrinsicConfig) -> torch.Tensor:
    del cfg
    return torch.ones_like(success.float())


INTRINSIC_GATE_REGISTRY = {
    "failure_only": _gate_failure_only,
    "asymmetric": _gate_asymmetric,
    "none": _gate_none,
}


def compose_total_advantage(
    *,
    external_advantages: torch.Tensor,
    intrinsic_token_advantages: torch.Tensor,
    gate: torch.Tensor,
    eta: float,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compose GRPO total token advantage with masking.

    A_total(i,t) = A_ext(i,t) + eta * g_i * A_int(i,t)
    """
    if external_advantages.shape != intrinsic_token_advantages.shape:
        raise ValueError(
            "external_advantages and intrinsic_token_advantages must have identical shape, "
            f"got {tuple(external_advantages.shape)} vs {tuple(intrinsic_token_advantages.shape)}."
        )
    if response_mask.shape != external_advantages.shape:
        raise ValueError(
            "response_mask must match advantage shape, "
            f"got mask {tuple(response_mask.shape)} vs advantages {tuple(external_advantages.shape)}."
        )
    if gate.ndim != 1 or gate.size(0) != external_advantages.size(0):
        raise ValueError(
            "gate must be 1D with batch dimension size, "
            f"got gate {tuple(gate.shape)} for batch size {external_advantages.size(0)}."
        )
    total = external_advantages + float(eta) * gate.unsqueeze(-1).to(external_advantages) * intrinsic_token_advantages
    return total * response_mask.float()


def compute_intrinsic_gate(
    *,
    extrinsic_scores: torch.Tensor,
    response_mask: torch.Tensor,
    cfg: IntrinsicConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    response_mask_f = response_mask.float()
    extrinsic_final = (extrinsic_scores * response_mask_f).sum(dim=-1)
    # Verifiable reasoning setup assumes binary correctness externally.
    success = extrinsic_final > cfg.success_threshold
    gate_fn = INTRINSIC_GATE_REGISTRY[cfg.gate_mode]
    gate = gate_fn(success, cfg)
    if gate.ndim == 0:
        gate = gate.expand_as(extrinsic_final)
    return gate, extrinsic_final, success


def apply_intrinsic_rule(
    *,
    ssl_error: torch.Tensor,
    response_mask: torch.Tensor,
    extrinsic_scores: torch.Tensor,
    cfg: IntrinsicConfig,
    stats: RunningZScore,
    intrinsic_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transform AR error into token-level intrinsic advantage basis [B, T]."""
    response_mask_f = response_mask.float()
    if intrinsic_mask is None:
        intrinsic_mask_f = response_mask_f
    else:
        intrinsic_mask_f = intrinsic_mask.float()
    raw = ssl_error * intrinsic_mask_f
    smoothed = _causal_moving_average(
        values=raw,
        mask=intrinsic_mask_f,
        window=cfg.temporal_smoothing_window,
        epsilon=cfg.epsilon,
    )

    if cfg.normalize_scope == "running_zscore":
        stats.update(smoothed[intrinsic_mask_f.bool()])
    normalized = _normalize(smoothed, cfg=cfg, stats=stats, mask=intrinsic_mask_f)

    transformed = _transform_intrinsic(normalized, cfg=cfg)

    if cfg.token_mode == "dense":
        intrinsic = transformed
    elif cfg.token_mode == "last_token":
        seq_signal = (transformed * intrinsic_mask_f).sum(dim=-1) / intrinsic_mask_f.sum(dim=-1).clamp_min(1.0)
        intrinsic = torch.zeros_like(normalized)
        lengths = response_mask_f.sum(dim=-1).long().clamp_min(1)
        last_idx = lengths - 1
        intrinsic[torch.arange(intrinsic.size(0)), last_idx] = seq_signal
    else:
        raise ValueError(f"Unsupported intrinsic token_mode: {cfg.token_mode}")

    intrinsic = torch.clamp(intrinsic, min=-cfg.clip_value, max=cfg.clip_value)
    intrinsic = intrinsic * intrinsic_mask_f

    metric_mask = intrinsic_mask_f.bool()
    if metric_mask.any():
        raw_vals = raw[metric_mask]
        smooth_vals = smoothed[metric_mask]
        z_vals = normalized[metric_mask]
        transformed_vals = transformed[metric_mask]
        intrinsic_vals = intrinsic[metric_mask]
        raw_mean = raw_vals.mean().item()
        raw_std = raw_vals.std(unbiased=False).item()
        smooth_mean = smooth_vals.mean().item()
        smooth_std = smooth_vals.std(unbiased=False).item()
        intrinsic_mean = intrinsic_vals.mean().item()
        intrinsic_max = intrinsic_vals.max().item()
        intrinsic_min = intrinsic_vals.min().item()
        z_mean = z_vals.mean().item()
        z_std = z_vals.std(unbiased=False).item()
        transformed_mean = transformed_vals.mean().item()
        transformed_std = transformed_vals.std(unbiased=False).item()
    else:
        raw_mean = 0.0
        raw_std = 0.0
        smooth_mean = 0.0
        smooth_std = 0.0
        intrinsic_mean = 0.0
        intrinsic_max = 0.0
        intrinsic_min = 0.0
        z_mean = 0.0
        z_std = 0.0
        transformed_mean = 0.0
        transformed_std = 0.0

    gate, _, success = compute_intrinsic_gate(
        extrinsic_scores=extrinsic_scores,
        response_mask=response_mask,
        cfg=cfg,
    )
    metrics = {
        "research/intrinsic/raw_mean": raw_mean,
        "research/intrinsic/raw_std": raw_std,
        "research/intrinsic/smoothed_mean": smooth_mean,
        "research/intrinsic/smoothed_std": smooth_std,
        "research/intrinsic/success_rate": success.float().mean().item(),
        "research/intrinsic/gate_mean": gate.float().mean().item(),
        "research/intrinsic/mean": intrinsic_mean,
        "research/intrinsic/max": intrinsic_max,
        "research/intrinsic/min": intrinsic_min,
        "research/intrinsic/z_mean": z_mean,
        "research/intrinsic/z_std": z_std,
        "research/intrinsic/transformed_mean": transformed_mean,
        "research/intrinsic/transformed_std": transformed_std,
    }
    return intrinsic, metrics
