#!/usr/bin/env python3
"""Merge per-split AR evaluation outputs into one directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / 'merged'
    result_files = sorted(input_dir.glob('split_*/split_results.jsonl'))
    summary_files = sorted(input_dir.glob('split_*/summary.json'))
    if not result_files:
        raise FileNotFoundError(f'No split_*/split_results.jsonl found in {input_dir}')
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f'{output_dir} exists; use --overwrite')
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = []
    for path in result_files:
        merged.extend(load_jsonl(path))
    merged.sort(key=lambda r: (int(r.get('split_id', -1)), str(r.get('partition', '')), str(r.get('question_uid', '')), int(r.get('sample_idx', 0)), str(r.get('uid', ''))))
    (output_dir / 'split_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in merged), encoding='utf-8')

    summaries = [json.loads(path.read_text(encoding='utf-8')) for path in summary_files]
    summary = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'num_split_dirs': len(result_files),
        'merged_rows': len(merged),
        'source_summaries': summaries,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
