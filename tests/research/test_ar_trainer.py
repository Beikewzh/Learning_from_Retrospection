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

import os

import torch

from research.ar.model import TinyLatentAR
from research.ar.scorer import ARScorer
from research.ar.trainer import ARTrainer
from research.config import ARConfig


def _make_sequences(num_samples: int = 64, seq_len: int = 16, hidden_dim: int = 8):
    torch.manual_seed(0)
    sequences = []
    drift = torch.linspace(-0.2, 0.2, steps=hidden_dim)
    for _ in range(num_samples):
        x = torch.zeros(seq_len, hidden_dim)
        x[0] = torch.randn(hidden_dim) * 0.1
        for t in range(1, seq_len):
            x[t] = 0.9 * x[t - 1] + drift
        sequences.append(x)
    return sequences


@torch.no_grad()
def _sequence_mse(model: TinyLatentAR, sequences: list[torch.Tensor]) -> float:
    losses = []
    for seq in sequences:
        x = seq[:-1].unsqueeze(0)
        y = seq[1:].unsqueeze(0)
        pred = model(x)
        losses.append((pred - y).pow(2).mean().item())
    return float(sum(losses) / max(len(losses), 1))


def test_ar_trainer_improves_synthetic_loss(tmp_path):
    sequences = _make_sequences()
    cfg = ARConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        lr=5e-4,
        batch_size=16,
        train_steps=80,
        train_every_n_steps=1,
        min_buffer_samples=16,
        device="cpu",
        max_seq_len=64,
        num_workers=0,
    )

    baseline_model = TinyLatentAR(
        latent_dim=sequences[0].size(1),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
    )
    baseline_loss = _sequence_mse(baseline_model, sequences)

    trainer = ARTrainer(config=cfg, work_dir=str(tmp_path))
    out = trainer.train_from_sequences(sequences=sequences, global_step=10)
    assert out is not None
    assert os.path.exists(out.checkpoint_path)

    scorer = ARScorer.load_from_checkpoint(
        checkpoint_path=os.path.join(tmp_path, "checkpoints", "latest.pt"),
        device="cpu",
    )
    trained_loss = _sequence_mse(scorer.model, sequences)

    assert trained_loss < baseline_loss
