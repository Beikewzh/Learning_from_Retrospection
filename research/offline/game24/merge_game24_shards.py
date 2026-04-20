#!/usr/bin/env python3
"""Merge sharded Game24 offline latent-collection outputs into one directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=str, required=True, help="Root containing shard_XX subdirectories.")
    parser.add_argument("--output-dir", type=str, default=None, help="Merged output directory. Default: <input-root>/merged")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "merged"
    shard_dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("shard_")])

    if not shard_dirs:
        single_metadata = input_root / "metadata.jsonl"
        single_buffer = input_root / "buffer" / "shards"
        if single_metadata.exists() and single_buffer.exists():
            shard_dirs = [input_root]
        else:
            raise FileNotFoundError(f"No shard directories found under {input_root}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    buffer_shard_dir = output_dir / "buffer" / "shards"
    buffer_shard_dir.mkdir(parents=True, exist_ok=True)

    merged_metadata: list[dict] = []
    seen_uids: set[str] = set()
    total_samples = 0
    total_shards = 0
    next_shard_idx = 0
    shard_summaries = []

    for shard_dir in shard_dirs:
        metadata_path = shard_dir / "metadata.jsonl"
        summary_path = shard_dir / "collection_summary.json"
        local_shards = sorted((shard_dir / "buffer" / "shards").glob("part-*.parquet"))

        if metadata_path.exists():
            for row in load_jsonl(metadata_path):
                uid = str(row["uid"])
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                merged_metadata.append(row)

        for shard_path in local_shards:
            target = buffer_shard_dir / f"part-{next_shard_idx:06d}.parquet"
            shutil.copy2(shard_path, target)
            next_shard_idx += 1
            total_shards += 1

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            shard_summaries.append(summary)
            total_samples += int(summary.get("buffer_stats", {}).get("buffer/total_samples", 0))

    merged_metadata.sort(key=lambda row: (int(row.get("step", 0)), str(row.get("uid", ""))))
    (output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_metadata),
        encoding="utf-8",
    )

    manifest = {
        "total_samples": total_samples,
        "total_shards": total_shards,
        "next_shard_idx": next_shard_idx,
    }
    (output_dir / "buffer" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    merged_summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "num_input_shards": len(shard_dirs),
        "merged_metadata_rows": len(merged_metadata),
        "buffer_stats": {
            "buffer/total_samples": total_samples,
            "buffer/total_shards": total_shards,
            "buffer/pending_rows": 0,
        },
        "shard_dirs": [str(p) for p in shard_dirs],
        "source_summaries": shard_summaries,
    }
    (output_dir / "collection_summary.json").write_text(
        json.dumps(merged_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(merged_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
