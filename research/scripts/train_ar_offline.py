#!/usr/bin/env python3
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

"""Train AR model from latent buffer without running RL loop."""

from __future__ import annotations

import argparse
import os

from research.ar import ARTrainer
from research.buffer import ParquetLatentBuffer
from research.config import ARConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--min-buffer-samples", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buffer = ParquetLatentBuffer(
        root_dir=os.path.abspath(args.buffer_dir),
        shard_max_samples=1024,
        compression="zstd",
        max_disk_gb=1000.0,
    )
    sequences = buffer.load_sequences(max_samples=args.max_samples, seed=args.global_step)

    cfg = ARConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        min_buffer_samples=args.min_buffer_samples,
        device=args.device,
        max_seq_len=args.max_seq_len,
        train_every_n_steps=1,
    )
    trainer = ARTrainer(config=cfg, work_dir=os.path.abspath(args.output_dir))
    out = trainer.train_from_sequences(sequences=sequences, global_step=args.global_step)
    if out is None:
        print("Not enough samples to train AR model.")
    else:
        print(f"Saved checkpoint: {out.checkpoint_path}")
        print(f"Train loss: {out.train_loss:.6f} in {out.steps} steps")


if __name__ == "__main__":
    main()
