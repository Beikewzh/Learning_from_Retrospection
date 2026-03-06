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

"""Tiny causal Transformer for next-latent prediction."""

from __future__ import annotations

import torch
from torch import nn


class TinyLatentAR(nn.Module):
    def __init__(self, latent_dim: int, d_model: int, n_layers: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, H]
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, H], got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.in_proj(x) + self.pos_emb(pos)

        # causal mask: prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.out_proj(h)
