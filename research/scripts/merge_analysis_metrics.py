#!/usr/bin/env python3
"""Merge per-shard analysis metric outputs into one directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_repo_mounted_path(args.input_dir)
    output_dir = resolve_repo_mounted_path(args.output_dir) if args.output_dir else input_dir / 'merged'
    metric_files = sorted(input_dir.glob('metrics_shard_*.jsonl'))
    summary_files = sorted(input_dir.glob('metrics_shard_*_summary.json'))
    if not metric_files:
        raise FileNotFoundError(f'No metrics_shard_*.jsonl found in {input_dir}')
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f'{output_dir} exists; use --overwrite')
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = []
    seen = set()
    for path in metric_files:
        for row in load_jsonl(path):
            uid = row['uid']
            if uid in seen:
                continue
            seen.add(uid)
            merged.append(row)
    merged.sort(key=lambda r: (str(r.get('question_uid', '')), int(r.get('sample_idx', 0)), str(r['uid'])))
    (output_dir / 'metrics.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in merged),
        encoding='utf-8',
    )

    source_summaries = []
    for path in summary_files:
        source_summaries.append(json.loads(path.read_text(encoding='utf-8')))
    summary = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'num_metric_files': len(metric_files),
        'merged_rows': len(merged),
        'source_summaries': source_summaries,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
