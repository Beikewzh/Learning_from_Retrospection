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

"""Offline trainer for tiny latent AR model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from research.ar.model import TinyLatentAR
from research.config import ARConfig


class _LatentSequenceDataset(Dataset):
    def __init__(self, sequences: list[torch.Tensor]):
        self.samples = [seq.float() for seq in sequences if seq.ndim == 2 and seq.size(0) > 1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class _LatentSequencePathDataset(Dataset):
    def __init__(self, sequence_paths: list[str | os.PathLike[str]]):
        self.samples = [str(Path(path)) for path in sequence_paths]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = torch.load(self.samples[idx], map_location="cpu", weights_only=False)
        if not isinstance(seq, torch.Tensor):
            raise TypeError(f"Expected tensor in {self.samples[idx]}, got {type(seq)!r}")
        if seq.ndim != 2 or seq.size(0) <= 1:
            raise ValueError(f"Invalid latent sequence in {self.samples[idx]} with shape {tuple(seq.shape)}")
        return seq.float()


def _collate_sequences(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = [x.size(0) - 1 for x in batch]  # next-step target length
    hidden_dim = batch[0].size(1)
    max_len = max(lengths)

    x = torch.zeros(len(batch), max_len, hidden_dim, dtype=torch.float32)
    y = torch.zeros(len(batch), max_len, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, seq in enumerate(batch):
        inp = seq[:-1]
        tgt = seq[1:]
        t = inp.size(0)
        x[i, :t] = inp
        y[i, :t] = tgt
        mask[i, :t] = True

    # Transformer expects True for padded positions in key_padding_mask
    key_padding_mask = ~mask
    return {"x": x, "y": y, "mask": mask, "key_padding_mask": key_padding_mask}


@dataclass
class ARTrainOutput:
    checkpoint_path: str
    train_loss: float
    steps: int


class ARTrainer:
    def __init__(self, config: ARConfig, work_dir: str):
        self.config = config
        self.work_dir = os.path.abspath(work_dir)
        self.ckpt_dir = os.path.join(self.work_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _build_model(self, latent_dim: int, device: torch.device) -> TinyLatentAR:
        return TinyLatentAR(
            latent_dim=latent_dim,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
        ).to(device)

    def train_from_sequences(self, sequences: list[torch.Tensor], global_step: int) -> ARTrainOutput | None:
        if len(sequences) < self.config.min_buffer_samples:
            return None

        dataset = _LatentSequenceDataset(sequences)
        if len(dataset) < self.config.min_buffer_samples:
            return None

        device = torch.device(self.config.device)
        latent_dim = dataset[0].size(1)
        model = self._build_model(latent_dim=latent_dim, device=device)
        model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_sequences,
        )
        data_iter = iter(dataloader)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        running_loss = 0.0
        seen_steps = 0

        for _ in range(self.config.train_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)

            pred = model(x, key_padding_mask=key_padding_mask)
            per_token = (pred - y).pow(2).mean(dim=-1)
            denom = mask.float().sum().clamp_min(1.0)
            loss = (per_token * mask.float()).sum() / denom

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.detach().item()
            seen_steps += 1

        mean_loss = running_loss / max(seen_steps, 1)
        ckpt_name = f"ar_step_{global_step}.pt"
        tmp_path = os.path.join(self.ckpt_dir, ckpt_name + ".tmp")
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        latest_path = os.path.join(self.ckpt_dir, "latest.pt")

        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "config": self.config.__dict__,
            "global_step": global_step,
            "train_loss": mean_loss,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        torch.save(payload, latest_path)

        return ARTrainOutput(checkpoint_path=ckpt_path, train_loss=mean_loss, steps=seen_steps)

    def train_from_sequence_paths(self, sequence_paths: list[str | os.PathLike[str]], global_step: int) -> ARTrainOutput | None:
        if len(sequence_paths) < self.config.min_buffer_samples:
            return None

        dataset = _LatentSequencePathDataset(sequence_paths)
        if len(dataset) < self.config.min_buffer_samples:
            return None

        device = torch.device(self.config.device)
        latent_dim = dataset[0].size(1)
        model = self._build_model(latent_dim=latent_dim, device=device)
        model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate_sequences,
        )
        data_iter = iter(dataloader)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        running_loss = 0.0
        seen_steps = 0

        for _ in range(self.config.train_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)

            pred = model(x, key_padding_mask=key_padding_mask)
            per_token = (pred - y).pow(2).mean(dim=-1)
            denom = mask.float().sum().clamp_min(1.0)
            loss = (per_token * mask.float()).sum() / denom

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.detach().item()
            seen_steps += 1

        mean_loss = running_loss / max(seen_steps, 1)
        ckpt_name = f"ar_step_{global_step}.pt"
        tmp_path = os.path.join(self.ckpt_dir, ckpt_name + ".tmp")
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        latest_path = os.path.join(self.ckpt_dir, "latest.pt")

        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "config": self.config.__dict__,
            "global_step": global_step,
            "train_loss": mean_loss,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        torch.save(payload, latest_path)

        return ARTrainOutput(checkpoint_path=ckpt_path, train_loss=mean_loss, steps=seen_steps)
