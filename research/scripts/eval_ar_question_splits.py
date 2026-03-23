#!/usr/bin/env python3
"""Evaluate AR error across non-overlapping question-level CV folds.

Reuses precomputed metrics (lengths, decay) and only retrains/scores AR per fold.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ar import ARScorer, ARTrainer
from research.buffer import ParquetLatentBuffer
from research.config import ARConfig
from research.latent.codec import deserialize_latent_tensor


def resolve_repo_mounted_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    p_str = str(p)
    if p.exists():
        return p.resolve()

    host_repo = os.environ.get('REPO_ROOT')
    if host_repo:
        try:
            host_repo_path = Path(host_repo).expanduser().resolve()
            host_repo_str = str(host_repo_path)
        except Exception:
            host_repo_str = ''
        if host_repo_str and (p_str == host_repo_str or p_str.startswith(host_repo_str + '/')):
            rel_str = p_str[len(host_repo_str):].lstrip('/')
            return PROJECT_ROOT / rel_str

    if p_str.startswith('/workspace/') or p_str == '/workspace':
        return p
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--metrics-path', type=str, default='')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--use-reasoning-span', action='store_true')
    parser.add_argument('--min-seq-len', type=int, default=4)
    parser.add_argument('--n-splits', type=int, default=5, help='Number of non-overlapping question folds')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split-offset', type=int, default=0)
    parser.add_argument('--single-split', action='store_true', help='Evaluate only the fold given by --split-offset')
    parser.add_argument('--ar-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--ar-max-samples', type=int, default=1000)
    parser.add_argument('--ar-train-steps', type=int, default=100)
    parser.add_argument('--ar-batch-size', type=int, default=8)
    parser.add_argument('--ar-lr', type=float, default=1e-4)
    parser.add_argument('--ar-d-model', type=int, default=256)
    parser.add_argument('--ar-n-layers', type=int, default=2)
    parser.add_argument('--ar-n-heads', type=int, default=4)
    parser.add_argument('--ar-dropout', type=float, default=0.1)
    parser.add_argument('--ar-max-seq-len', type=int, default=4096)
    parser.add_argument('--ar-min-buffer-samples', type=int, default=64)
    parser.add_argument('--spill-train-sequences', action='store_true', help='Write train sequences to disk and train lazily from files')
    parser.add_argument('--keep-spilled-sequences', action='store_true', help='Keep spilled train sequence files after training')
    parser.add_argument('--overwrite', action='store_true')
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


@torch.no_grad()
def score_sequence_mean_error(seq: torch.Tensor, scorer: ARScorer) -> float | None:
    latents = seq.unsqueeze(0)
    response_mask = torch.ones((1, seq.shape[0]), dtype=torch.bool)
    errors = scorer.score(latents, response_mask=response_mask).detach().cpu().numpy()[0]
    valid = errors[1:]
    if valid.size == 0:
        return None
    return float(valid.mean())


def main() -> None:
    args = parse_args()
    run_dir = resolve_repo_mounted_path(args.run_dir)
    if args.metrics_path:
        metrics_path = resolve_repo_mounted_path(args.metrics_path)
    else:
        metrics_path = run_dir / 'spectrum_parallel' / 'merged' / 'metrics.jsonl'
    output_dir = resolve_repo_mounted_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'split_results.jsonl'
    summary_path = output_dir / 'summary.json'
    if results_path.exists() and not args.overwrite:
        raise FileExistsError(f'{results_path} exists; use --overwrite')

    metadata_path = run_dir / 'metadata.jsonl'
    buffer_dir = run_dir / 'buffer'
    if not metadata_path.exists():
        raise FileNotFoundError(f'missing metadata: {metadata_path}')
    if not buffer_dir.exists():
        raise FileNotFoundError(f'missing buffer: {buffer_dir}')
    if not metrics_path.exists():
        raise FileNotFoundError(f'missing metrics: {metrics_path}')

    metadata_rows = load_jsonl(metadata_path)
    metadata_by_uid = {row['uid']: row for row in metadata_rows}
    metric_rows = load_jsonl(metrics_path)
    metrics_by_uid = {row['uid']: row for row in metric_rows}

    by_question: dict[str, list[dict]] = {}
    for row in metric_rows:
        by_question.setdefault(str(row['question_uid']), []).append(row)
    mixed_questions = sorted(
        qid for qid, rows in by_question.items()
        if any(bool(r['success']) for r in rows) and any((not bool(r['success'])) for r in rows)
    )
    if len(mixed_questions) < 2:
        raise RuntimeError('Need at least two mixed-outcome questions for train/held-out splits')

    buffer = ParquetLatentBuffer(
        root_dir=os.path.abspath(buffer_dir),
        shard_max_samples=256,
        compression='zstd',
        max_disk_gb=1000.0,
    )

    cfg = ARConfig(
        d_model=args.ar_d_model,
        n_layers=args.ar_n_layers,
        n_heads=args.ar_n_heads,
        dropout=args.ar_dropout,
        lr=args.ar_lr,
        batch_size=args.ar_batch_size,
        train_steps=args.ar_train_steps,
        min_buffer_samples=args.ar_min_buffer_samples,
        device=args.ar_device,
        max_seq_len=args.ar_max_seq_len,
        train_every_n_steps=1,
    )

    summaries = []
    with results_path.open('w', encoding='utf-8') as out_f:
        question_ids = mixed_questions[:]
        rng = random.Random(args.seed)
        rng.shuffle(question_ids)
        folds = [list(chunk) for chunk in np.array_split(np.asarray(question_ids, dtype=object), args.n_splits)]
        split_ids = [args.split_offset] if args.single_split else list(range(args.split_offset, args.split_offset + args.n_splits))
        for split_id in split_ids:
            fold_idx = split_id % args.n_splits
            heldout_questions = set(str(x) for x in folds[fold_idx])
            train_questions = set(question_ids) - heldout_questions
            if not heldout_questions or not train_questions:
                raise RuntimeError(f'Invalid fold split for fold {fold_idx}')
            question_partition = {qid: 'train' for qid in train_questions}
            question_partition.update({qid: 'heldout' for qid in heldout_questions})

            train_sequences: list[torch.Tensor] = []
            train_sequence_paths: list[str] = []
            train_seen = 0
            train_short = 0
            spill_dir = output_dir / f'split_{split_id:02d}_train_sequences'
            if args.spill_train_sequences:
                if spill_dir.exists() and args.overwrite:
                    shutil.rmtree(spill_dir)
                spill_dir.mkdir(parents=True, exist_ok=True)
            for row in buffer.iter_rows():
                uid = row['uid']
                meta = metadata_by_uid.get(uid)
                metrics = metrics_by_uid.get(uid)
                if meta is None or metrics is None:
                    continue
                qid = str(metrics['question_uid'])
                if qid not in train_questions:
                    continue
                seq = extract_sequence_for_analysis(row, meta, use_reasoning_span=args.use_reasoning_span)
                if seq.shape[0] < args.min_seq_len:
                    train_short += 1
                    del seq
                    continue
                train_seen += 1
                if args.spill_train_sequences:
                    seq_path = spill_dir / f'{len(train_sequence_paths):06d}.pt'
                    torch.save(seq.to(dtype=torch.float16).cpu(), seq_path)
                    train_sequence_paths.append(str(seq_path))
                    del seq
                else:
                    train_sequences.append(seq)
                current_train_count = len(train_sequence_paths) if args.spill_train_sequences else len(train_sequences)
                if current_train_count >= args.ar_max_samples:
                    break

            trainer = ARTrainer(config=cfg, work_dir=os.path.abspath(output_dir / f'split_{split_id:02d}_ar'))
            if args.spill_train_sequences:
                gc.collect()
                train_out = trainer.train_from_sequence_paths(train_sequence_paths, global_step=split_id)
            else:
                train_out = trainer.train_from_sequences(train_sequences, global_step=split_id)
            if train_out is None:
                raise RuntimeError(f'Not enough samples to train AR for split {split_id}')
            scorer = ARScorer.load_from_checkpoint(train_out.checkpoint_path, device=args.ar_device)
            del train_sequences
            gc.collect()
            if args.ar_device == 'cuda':
                torch.cuda.empty_cache()

            scored_rows = 0
            scored_short = 0
            for row in buffer.iter_rows():
                uid = row['uid']
                meta = metadata_by_uid.get(uid)
                metrics = metrics_by_uid.get(uid)
                if meta is None or metrics is None:
                    continue
                qid = str(metrics['question_uid'])
                partition = question_partition.get(qid)
                if partition is None:
                    continue
                seq = extract_sequence_for_analysis(row, meta, use_reasoning_span=args.use_reasoning_span)
                if seq.shape[0] < args.min_seq_len:
                    scored_short += 1
                    del seq
                    continue
                ar_error = score_sequence_mean_error(seq, scorer)
                del seq
                if ar_error is None:
                    continue
                result = {
                    'split_id': split_id,
                'fold_index': fold_idx,
                    'split_seed': args.seed + split_id,
                    'partition': partition,
                    'uid': uid,
                    'question_uid': qid,
                    'sample_idx': int(metrics.get('sample_idx', 0)),
                    'success': bool(metrics['success']),
                    'score_accuracy': float(metrics['score_accuracy']),
                    'response_length': int(metrics['response_length']),
                    'analysis_length': int(metrics['analysis_length']),
                    'decay_rate': float(metrics['decay_rate']),
                    'ar_error': float(ar_error),
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                scored_rows += 1

            summary = {
                'split_id': split_id,
                'fold_index': fold_idx,
                'split_seed': args.seed + split_id,
                'mixed_questions_total': len(mixed_questions),
                'train_questions': len(train_questions),
                'heldout_questions': len(heldout_questions),
                'train_seen': train_seen,
                'train_short': train_short,
                'train_sequences': len(train_sequence_paths) if args.spill_train_sequences else min(train_seen - train_short, args.ar_max_samples),
                'spill_train_sequences': bool(args.spill_train_sequences),
                'train_checkpoint_path': train_out.checkpoint_path,
                'train_loss': train_out.train_loss,
                'scored_rows': scored_rows,
                'scored_short': scored_short,
            }
            summaries.append(summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            if args.spill_train_sequences and (not args.keep_spilled_sequences) and spill_dir.exists():
                shutil.rmtree(spill_dir)

    overall = {
        'run_dir': str(run_dir),
        'metrics_path': str(metrics_path),
        'output_dir': str(output_dir),
        'mixed_questions_total': len(mixed_questions),
        'n_splits': args.n_splits,
        'split_offset': args.split_offset,
        'use_reasoning_span': args.use_reasoning_span,
        'config': vars(args),
        'split_summaries': summaries,
    }
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'results_path': str(results_path), 'summary_path': str(summary_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
