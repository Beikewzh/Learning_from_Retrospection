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

"""Inference-time scorer that converts AR prediction errors into per-token SSL signal."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from research.ar.model import TinyLatentAR


@dataclass
class ScorerMeta:
    checkpoint_path: str
    global_step: int
    train_loss: float


class ARScorer:
    def __init__(self, model: TinyLatentAR, device: torch.device, meta: ScorerMeta):
        self.model = model
        self.device = device
        self.meta = meta
        self.model.eval()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str) -> "ARScorer":
        map_location = torch.device(device)
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        cfg = payload["config"]
        model = TinyLatentAR(
            latent_dim=int(payload["latent_dim"]),
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            dropout=float(cfg["dropout"]),
            max_seq_len=int(cfg["max_seq_len"]),
        ).to(map_location)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        meta = ScorerMeta(
            checkpoint_path=os.path.abspath(checkpoint_path),
            global_step=int(payload.get("global_step", -1)),
            train_loss=float(payload.get("train_loss", 0.0)),
        )
        return cls(model=model, device=map_location, meta=meta)

    @torch.no_grad()
    def score(self, latents: torch.Tensor, response_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return per-token prediction error with shape [B, T]."""
        if latents.ndim != 3:
            raise ValueError(f"Expected latents [B, T, H], got {tuple(latents.shape)}")
        latents = latents.to(self.device).float()
        bsz, seq_len, _ = latents.shape
        if seq_len <= 1:
            return torch.zeros((bsz, seq_len), device=latents.device, dtype=torch.float32)

        x = latents[:, :-1, :]
        y = latents[:, 1:, :]

        if response_mask is None:
            valid = torch.ones((bsz, seq_len - 1), device=latents.device, dtype=torch.bool)
        else:
            valid = response_mask.to(latents.device).bool()[:, 1:]

        key_padding_mask = ~valid
        pred = self.model(x, key_padding_mask=key_padding_mask)
        error = (pred - y).pow(2).mean(dim=-1)

        full = torch.zeros((bsz, seq_len), device=latents.device, dtype=torch.float32)
        full[:, 1:] = error
        if response_mask is not None:
            full = full * response_mask.to(latents.device).float()
        return full
