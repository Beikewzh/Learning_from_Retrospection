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
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--window-intervals", type=int, default=0)
    parser.add_argument("--interval-steps", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-job-type", type=str, default="offline_ar")
    parser.add_argument("--wandb-mode", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_intervals < 0:
        raise ValueError("--window-intervals must be >= 0")
    if args.interval_steps < 0:
        raise ValueError("--interval-steps must be >= 0")
    if args.window_intervals > 0 and args.interval_steps <= 0:
        raise ValueError("--interval-steps must be > 0 when --window-intervals > 0")

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"W&B unavailable ({exc}); continuing without W&B logging.")
        else:
            init_kwargs = {
                "project": args.wandb_project,
                "entity": args.wandb_entity or None,
                "name": args.wandb_run_name or None,
                "group": args.wandb_group or None,
                "job_type": args.wandb_job_type,
                "config": vars(args),
                "reinit": True,
            }
            if args.wandb_mode:
                init_kwargs["mode"] = args.wandb_mode
            wandb_run = wandb.init(**init_kwargs)

    buffer = ParquetLatentBuffer(
        root_dir=os.path.abspath(args.buffer_dir),
        shard_max_samples=1024,
        compression="zstd",
        max_disk_gb=1000.0,
    )
    window_min_step = None
    if args.window_intervals > 0:
        window_span = args.window_intervals * args.interval_steps
        window_min_step = max(0, args.global_step - window_span + 1)
    sequences = buffer.load_sequences(
        max_samples=args.max_samples,
        seed=args.global_step,
        min_step=window_min_step,
        max_step=args.global_step,
    )

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
        if wandb_run is not None:
            wandb_run.log(
                {
                    "offline_ar/status": 0,
                    "offline_ar/samples_loaded": len(sequences),
                    "offline_ar/min_buffer_samples": args.min_buffer_samples,
                    "offline_ar/window_intervals": args.window_intervals,
                    "offline_ar/window_min_step": window_min_step if window_min_step is not None else 0,
                }
            )
    else:
        print(f"Saved checkpoint: {out.checkpoint_path}")
        print(f"Train loss: {out.train_loss:.6f} in {out.steps} steps")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "offline_ar/status": 1,
                    "offline_ar/train_loss": out.train_loss,
                    "offline_ar/steps": out.steps,
                    "offline_ar/samples_loaded": len(sequences),
                    "offline_ar/global_step": args.global_step,
                    "offline_ar/window_intervals": args.window_intervals,
                    "offline_ar/window_min_step": window_min_step if window_min_step is not None else 0,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
