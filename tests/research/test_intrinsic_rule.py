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

import torch

from research.config import IntrinsicConfig
from research.intrinsic.reward_rule import (
    RunningZScore,
    apply_intrinsic_rule,
    compose_total_advantage,
    compute_intrinsic_gate,
)


def test_per_sequence_zscore_dense():
    ssl_error = torch.tensor([[1.0, 2.0, 3.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    extrinsic_scores = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    cfg = IntrinsicConfig(
        normalize_scope="per_sequence_zscore",
        normalize="per_sequence_zscore",
        clip_value=10.0,
        token_mode="dense",
    )
    stats = RunningZScore()
    intrinsic, metrics = apply_intrinsic_rule(
        ssl_error=ssl_error,
        response_mask=response_mask,
        extrinsic_scores=extrinsic_scores,
        cfg=cfg,
        stats=stats,
    )

    expected_first = torch.tensor([-1.2247, 0.0, 1.2247, 0.0])
    assert torch.allclose(intrinsic[0], expected_first, atol=1e-3)
    assert torch.allclose(intrinsic[1], torch.zeros(4), atol=1e-6)
    assert "research/intrinsic/raw_mean" in metrics
    assert "research/intrinsic/z_std" in metrics


def test_causal_smoothing_window_two():
    ssl_error = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    response_mask = torch.tensor([[1, 1, 1, 1]])
    extrinsic_scores = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    cfg = IntrinsicConfig(
        temporal_smoothing_window=2,
        normalize_scope="none",
        normalize="none",
        clip_value=10.0,
        token_mode="dense",
    )
    stats = RunningZScore()
    intrinsic, _ = apply_intrinsic_rule(
        ssl_error=ssl_error,
        response_mask=response_mask,
        extrinsic_scores=extrinsic_scores,
        cfg=cfg,
        stats=stats,
    )

    expected = torch.tensor([[1.0, 2.0, 4.0, 6.0]])
    assert torch.allclose(intrinsic, expected, atol=1e-5)


def test_compute_intrinsic_gate_variants():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    extrinsic_scores = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    cfg = IntrinsicConfig(gate_mode="failure_only", success_threshold=0.5)
    gate, _, success = compute_intrinsic_gate(
        extrinsic_scores=extrinsic_scores,
        response_mask=response_mask,
        cfg=cfg,
    )
    assert torch.allclose(gate, torch.tensor([0.0, 1.0]))
    assert torch.allclose(success.float(), torch.tensor([1.0, 0.0]))

    cfg = IntrinsicConfig(gate_mode="asymmetric", lambda_success_gate=0.2, success_threshold=0.5)
    gate, _, _ = compute_intrinsic_gate(
        extrinsic_scores=extrinsic_scores,
        response_mask=response_mask,
        cfg=cfg,
    )
    assert torch.allclose(gate, torch.tensor([-0.2, 1.0]))


def test_compose_total_advantage_respects_mask_and_gate():
    external = torch.tensor([[1.0, 1.0, 1.0], [-0.5, -0.5, -0.5]])
    intrinsic = torch.tensor([[2.0, 0.0, -2.0], [3.0, 3.0, 3.0]])
    gate = torch.tensor([0.0, 1.0])
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    total = compose_total_advantage(
        external_advantages=external,
        intrinsic_token_advantages=intrinsic,
        gate=gate,
        eta=0.1,
        response_mask=response_mask,
    )

    expected = torch.tensor([[1.0, 1.0, 0.0], [-0.2, 0.0, 0.0]])
    assert torch.allclose(total, expected, atol=1e-6)
