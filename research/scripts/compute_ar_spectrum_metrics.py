#!/usr/bin/env python3
"""Compute per-trajectory spectrum decay and AR error for one shard of an offline run."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ar import ARScorer
from research.buffer import ParquetLatentBuffer
from research.latent.codec import deserialize_latent_tensor
from research.scripts.offline_utils import resolve_repo_mounted_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ar-checkpoint', type=str, default='')
    parser.add_argument('--ar-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--use-reasoning-span', action='store_true')
    parser.add_argument('--max-components', type=int, default=256)
    parser.add_argument('--min-seq-len', type=int, default=4)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-ar', action='store_true')
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_sequence_for_analysis(row: dict, meta: dict, use_reasoning_span: bool):
    seq = deserialize_latent_tensor(
        row['latent_blob'],
        seq_len=int(row['response_length']),
        hidden_dim=int(row['hidden_dim']),
        dtype=str(row['dtype']),
    ).float()

    start = 0
    end = seq.shape[0]
    if use_reasoning_span and meta.get('think_token_start') is not None and meta.get('think_token_end') is not None:
        start = int(meta['think_token_start'])
        end = int(meta['think_token_end'])
        start = max(0, min(start, seq.shape[0]))
        end = max(start + 1, min(end, seq.shape[0]))
    return seq[start:end]


def covariance_eigenspectrum(seq):
    x = seq.numpy().astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        return None
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    # eigvals = np.clip(eigvals, 1e-12, None)
    return eigvals


def estimate_powerlaw_exponent(eigvals, max_components=64):
    eigvals = np.asarray(eigvals, dtype=np.float64)
    k = min(len(eigvals), max_components)
    if k < 3:
        return None
    ranks = np.arange(1, k + 1, dtype=np.float64)
    x = np.log(ranks)
    y = np.log(eigvals[:k])
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    x = x[valid]
    y = y[valid]
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(-slope)


@torch.no_grad()
def score_sequence_mean_error(seq, scorer):
    latents = seq.unsqueeze(0)
    response_mask = torch.ones((1, seq.shape[0]), dtype=torch.bool)
    errors = scorer.score(latents, response_mask=response_mask).detach().cpu().numpy()[0]
    valid = errors[1:]
    if valid.size == 0:
        return None
    return float(valid.mean())


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError('--num-shards must be > 0')
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError('--shard-index must be in [0, num_shards)')

    if not args.skip_ar and not args.ar_checkpoint:
        raise ValueError('--ar-checkpoint is required unless --skip-ar is set')

    run_dir = resolve_repo_mounted_path(args.run_dir)
    output_dir = resolve_repo_mounted_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f'metrics_shard_{args.shard_index:02d}.jsonl'
    summary_path = output_dir / f'metrics_shard_{args.shard_index:02d}_summary.json'
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f'{metrics_path} exists; use --overwrite')

    metadata_rows = load_jsonl(run_dir / 'metadata.jsonl')
    metadata_by_uid = {row['uid']: row for row in metadata_rows}
    selected_uids = {
        row['uid']
        for idx, row in enumerate(metadata_rows)
        if idx % args.num_shards == args.shard_index
    }

    buffer = ParquetLatentBuffer(
        root_dir=os.path.abspath(run_dir / 'buffer'),
        shard_max_samples=256,
        compression='zstd',
        max_disk_gb=1000.0,
    )
    scorer = None
    if not args.skip_ar:
        scorer = ARScorer.load_from_checkpoint(str(resolve_repo_mounted_path(args.ar_checkpoint)), device=args.ar_device)

    processed = 0
    written = 0
    skipped_short = 0
    with metrics_path.open('w', encoding='utf-8') as out:
        for row in buffer.iter_rows():
            uid = row['uid']
            if uid not in selected_uids:
                continue
            meta = metadata_by_uid.get(uid)
            if meta is None:
                continue
            processed += 1
            seq = extract_sequence_for_analysis(row, meta, use_reasoning_span=args.use_reasoning_span)
            if seq.shape[0] < args.min_seq_len:
                skipped_short += 1
                del seq
                continue
            eigvals = covariance_eigenspectrum(seq)
            if eigvals is None:
                del seq
                continue
            decay_rate = estimate_powerlaw_exponent(eigvals, max_components=args.max_components)
            if decay_rate is None:
                del seq
                continue
            ar_error = None if scorer is None else score_sequence_mean_error(seq, scorer)
            record = {
                'uid': uid,
                'question_uid': meta.get('question_uid', meta['uid']),
                'sample_idx': int(meta.get('sample_idx', 0)),
                'success': bool(meta['success']),
                'score_accuracy': float(meta['score_accuracy']),
                'response_length': int(meta['response_length']),
                'analysis_length': int(seq.shape[0]),
                'decay_rate': float(decay_rate),
                'ar_error': None if ar_error is None else float(ar_error),
                'top_eigvals': [float(x) for x in eigvals[:10]],
                'shard_index': args.shard_index,
                'num_shards': args.num_shards,
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1
            del seq

    summary = {
        'run_dir': str(run_dir),
        'output_dir': str(output_dir),
        'metrics_path': str(metrics_path),
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'selected_uids': len(selected_uids),
        'processed_rows': processed,
        'written_rows': written,
        'skipped_short': skipped_short,
        'ar_checkpoint': None if args.skip_ar else str(resolve_repo_mounted_path(args.ar_checkpoint)),
        'ar_device': None if args.skip_ar else args.ar_device,
        'skip_ar': args.skip_ar,
        'use_reasoning_span': args.use_reasoning_span,
        'max_components': args.max_components,
        'min_seq_len': args.min_seq_len,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
