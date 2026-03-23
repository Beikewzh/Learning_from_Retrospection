#!/usr/bin/env python3
"""Train a tiny AR model on extracted trajectories from one offline latent run."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ar import ARTrainer
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
            host_repo_path = None
            host_repo_str = ''
        if host_repo_str and (p_str == host_repo_str or p_str.startswith(host_repo_str + '/')):
            rel_str = p_str[len(host_repo_str):].lstrip('/')
            candidate = PROJECT_ROOT / rel_str
            return candidate

    if p_str.startswith('/workspace/') or p_str == '/workspace':
        return p
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--use-reasoning-span', action='store_true')
    parser.add_argument('--mixed-questions-only', action='store_true', help='Train AR only on question_uid groups that contain both success and failure')
    parser.add_argument('--max-samples', type=int, default=16000)
    parser.add_argument('--min-seq-len', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--train-steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-seq-len', type=int, default=4096)
    parser.add_argument('--min-buffer-samples', type=int, default=64)
    parser.add_argument('--spill-train-sequences', action='store_true')
    parser.add_argument('--keep-spilled-sequences', action='store_true')
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


def main() -> None:
    args = parse_args()
    run_dir = resolve_repo_mounted_path(args.run_dir)
    output_dir = resolve_repo_mounted_path(args.output_dir)
    metadata_path = run_dir / 'metadata.jsonl'
    buffer_dir = run_dir / 'buffer'

    if not metadata_path.exists():
        raise FileNotFoundError(f'missing metadata: {metadata_path}')
    if not buffer_dir.exists():
        raise FileNotFoundError(f'missing buffer: {buffer_dir}')

    metadata_rows = load_jsonl(metadata_path)
    metadata_by_uid = {row['uid']: row for row in metadata_rows}
    mixed_question_uids: set[str] | None = None
    if args.mixed_questions_only:
        by_question: dict[str, list[dict]] = {}
        for row in metadata_rows:
            by_question.setdefault(str(row.get('question_uid', row['uid'])), []).append(row)
        mixed_question_uids = {
            qid for qid, rows in by_question.items()
            if any(bool(r.get('success', False)) for r in rows) and any((not bool(r.get('success', False))) for r in rows)
        }

    buffer = ParquetLatentBuffer(
        root_dir=os.path.abspath(buffer_dir),
        shard_max_samples=256,
        compression='zstd',
        max_disk_gb=1000.0,
    )

    sequences = []
    sequence_paths: list[str] = []
    seen_rows = 0
    skipped_short = 0
    spill_dir = output_dir / 'train_sequences'
    if args.spill_train_sequences:
        if spill_dir.exists():
            shutil.rmtree(spill_dir)
        spill_dir.mkdir(parents=True, exist_ok=True)
    for row in buffer.iter_rows():
        meta = metadata_by_uid.get(row['uid'])
        if meta is None:
            continue
        if mixed_question_uids is not None:
            qid = str(meta.get('question_uid', meta['uid']))
            if qid not in mixed_question_uids:
                continue
        seen_rows += 1
        seq = extract_sequence_for_analysis(row, meta, use_reasoning_span=args.use_reasoning_span)
        if seq.shape[0] < args.min_seq_len:
            skipped_short += 1
            del seq
            continue
        if args.spill_train_sequences:
            seq_path = spill_dir / f'{len(sequence_paths):06d}.pt'
            torch.save(seq.to(dtype=torch.float16).cpu(), seq_path)
            sequence_paths.append(str(seq_path))
            del seq
        else:
            sequences.append(seq)
        current_count = len(sequence_paths) if args.spill_train_sequences else len(sequences)
        if current_count >= args.max_samples:
            break

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
    trainer = ARTrainer(config=cfg, work_dir=os.path.abspath(output_dir))
    if args.spill_train_sequences:
        gc.collect()
        out = trainer.train_from_sequence_paths(sequence_paths=sequence_paths, global_step=0)
    else:
        out = trainer.train_from_sequences(sequences=sequences, global_step=0)
    if out is None:
        raise RuntimeError('Not enough samples to train AR model')
    del sequences
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    summary = {
        'run_dir': str(run_dir),
        'output_dir': str(output_dir),
        'use_reasoning_span': args.use_reasoning_span,
        'mixed_questions_only': bool(args.mixed_questions_only),
        'device': args.device,
        'seen_rows': seen_rows,
        'skipped_short': skipped_short,
        'train_sequences': len(sequence_paths) if args.spill_train_sequences else current_count,
        'spill_train_sequences': bool(args.spill_train_sequences),
        'mixed_question_count': None if mixed_question_uids is None else len(mixed_question_uids),
        'checkpoint_path': out.checkpoint_path,
        'train_loss': out.train_loss,
        'steps': out.steps,
        'config': vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'train_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.spill_train_sequences and (not args.keep_spilled_sequences) and spill_dir.exists():
        shutil.rmtree(spill_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
