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
from research.intrinsic.reward_rule import RunningZScore, apply_intrinsic_rule


def test_weighted_additive_outcome_gating_dense():
    ssl_error = torch.tensor([[1.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    extrinsic_scores = torch.tensor([[0.4, 0.4, 0.0], [0.0, -0.2, -0.2]])

    cfg = IntrinsicConfig(
        mode="weighted_additive",
        outcome_gated=True,
        success_threshold=0.0,
        lambda_success=0.5,
        lambda_failure=0.25,
        normalize="none",
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

    expected = torch.tensor([[-0.5, -1.0, 0.0], [0.25, 0.25, 0.25]])
    assert torch.allclose(intrinsic, expected)
    assert 0.0 <= metrics["research/intrinsic/success_rate"] <= 1.0


def test_last_token_mode_and_clipping():
    ssl_error = torch.tensor([[10.0, 10.0, 10.0]])
    response_mask = torch.tensor([[1, 1, 1]])
    extrinsic_scores = torch.tensor([[-1.0, 0.0, 0.0]])

    cfg = IntrinsicConfig(
        mode="weighted_additive",
        outcome_gated=True,
        success_threshold=0.0,
        lambda_success=1.0,
        lambda_failure=1.0,
        normalize="none",
        clip_value=2.0,
        token_mode="last_token",
    )
    stats = RunningZScore()
    intrinsic, _ = apply_intrinsic_rule(
        ssl_error=ssl_error,
        response_mask=response_mask,
        extrinsic_scores=extrinsic_scores,
        cfg=cfg,
        stats=stats,
    )

    assert torch.allclose(intrinsic, torch.tensor([[0.0, 0.0, 2.0]]))
