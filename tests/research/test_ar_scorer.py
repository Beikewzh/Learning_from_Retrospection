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

from research.ar.model import TinyLatentAR
from research.ar.scorer import ARScorer, ScorerMeta


def _build_scorer() -> ARScorer:
    model = TinyLatentAR(
        latent_dim=4,
        d_model=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_seq_len=16,
    )
    return ARScorer(
        model=model,
        device=torch.device("cpu"),
        meta=ScorerMeta(checkpoint_path="unit-test", global_step=0, train_loss=0.0),
    )


def test_score_handles_rows_without_valid_transitions():
    scorer = _build_scorer()
    latents = torch.randn(2, 4, 4)
    response_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]])

    score = scorer.score(latents=latents, response_mask=response_mask)

    assert score.shape == (2, 4)
    assert torch.isfinite(score).all()
    assert torch.allclose(score[0], torch.zeros(4))
    assert score[1, 0].item() == 0.0
    assert score[1, 3].item() == 0.0
