#!/usr/bin/env python3
"""Export experiment_log.jsonl metrics into compact JSON/NPZ summaries.

This script reads EasyR1/LeaRS `experiment_log.jsonl` files and extracts a small
set of plotting-friendly metrics per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


DEFAULT_SEARCH_ROOTS = [
    Path("/scratch/p/psli/Learning_from_Retrospection_runs/checkpoints/tutorial"),
    Path("/scratch/p/psli/Learning_from_Retrospection/checkpoints/easy_r1"),
]


def get_nested(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def discover_logs(search_roots: list[Path], patterns: list[str]) -> list[Path]:
    logs: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("experiment_log.jsonl"):
            run_name = path.parent.name
            if patterns and not any(pattern in run_name for pattern in patterns):
                continue
            logs.append(path)
    return sorted(set(logs))


def summarize_run(log_path: Path) -> dict[str, Any]:
    steps: list[int] = []
    val_steps: list[int] = []
    val_accuracy: list[float] = []
    val_reward_score: list[float] = []
    val_response_length_mean: list[float] = []
    train_response_length_mean: list[float] = []
    train_accuracy: list[float] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = row.get("step")
            if isinstance(step, int):
                steps.append(step)

            train_acc = get_nested(row, "reward", "accuracy")
            if isinstance(train_acc, (int, float)):
                train_accuracy.append(float(train_acc))

            train_resp = get_nested(row, "response_length", "mean")
            if isinstance(train_resp, (int, float)):
                train_response_length_mean.append(float(train_resp))

            val_acc = get_nested(row, "val", "accuracy_reward")
            if isinstance(val_acc, (int, float)):
                val_steps.append(int(step) if isinstance(step, int) else len(val_steps))
                val_accuracy.append(float(val_acc))

                val_score = get_nested(row, "val", "reward_score")
                val_reward_score.append(float(val_score) if isinstance(val_score, (int, float)) else math.nan)

                val_resp = get_nested(row, "val_response_length", "mean")
                val_response_length_mean.append(
                    float(val_resp) if isinstance(val_resp, (int, float)) else math.nan
                )

    def clean(values: list[float]) -> list[float]:
        return [float(v) for v in values if not math.isnan(float(v))]

    val_accuracy_clean = clean(val_accuracy)
    train_resp_clean = clean(train_response_length_mean)
    val_resp_clean = clean(val_response_length_mean)

    def safe_stat(values: list[float], fn: str) -> float | None:
        if not values:
            return None
        if fn == "final":
            return float(values[-1])
        if fn == "min":
            return float(min(values))
        if fn == "max":
            return float(max(values))
        if fn == "mean":
            return float(statistics.fmean(values))
        if fn == "std":
            return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        raise ValueError(fn)

    return {
        "run_name": log_path.parent.name,
        "log_path": str(log_path),
        "max_step": max(steps) if steps else None,
        "num_records": len(steps),
        "num_val_points": len(val_steps),
        "val_steps": val_steps,
        "val_accuracy_reward": val_accuracy,
        "val_reward_score": val_reward_score,
        "val_response_length_mean": val_response_length_mean,
        "train_response_length_mean": train_response_length_mean,
        "train_accuracy_reward": train_accuracy,
        "final_val_accuracy_reward": safe_stat(val_accuracy_clean, "final"),
        "val_accuracy_reward_min": safe_stat(val_accuracy_clean, "min"),
        "val_accuracy_reward_max": safe_stat(val_accuracy_clean, "max"),
        "val_accuracy_reward_mean": safe_stat(val_accuracy_clean, "mean"),
        "val_accuracy_reward_std": safe_stat(val_accuracy_clean, "std"),
        "mean_train_response_length": safe_stat(train_resp_clean, "mean"),
        "mean_val_response_length": safe_stat(val_resp_clean, "mean"),
        "final_val_response_length": safe_stat(val_resp_clean, "final"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Directory to search recursively for experiment_log.jsonl files. Can be used multiple times.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Only include runs whose parent directory name contains this substring. Can be used multiple times.",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Explicit experiment_log.jsonl path. Can be used multiple times.",
    )
    parser.add_argument(
        "--output-prefix",
        default="metrics_export",
        help="Output prefix for <prefix>.json and <prefix>.npz",
    )
    args = parser.parse_args()

    search_roots = [Path(p) for p in args.search_root] if args.search_root else DEFAULT_SEARCH_ROOTS
    logs = [Path(p) for p in args.log]
    logs.extend(discover_logs(search_roots=search_roots, patterns=args.pattern))
    logs = sorted(set(logs))

    runs = [summarize_run(path) for path in logs]

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2)

    csv_path = output_prefix.with_suffix(".csv")
    scalar_fields = [
        "run_name",
        "log_path",
        "max_step",
        "num_records",
        "num_val_points",
        "final_val_accuracy_reward",
        "val_accuracy_reward_min",
        "val_accuracy_reward_max",
        "val_accuracy_reward_mean",
        "val_accuracy_reward_std",
        "mean_train_response_length",
        "mean_val_response_length",
        "final_val_response_length",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field) for field in scalar_fields})

    print(f"Wrote {len(runs)} runs to {json_path}")
    print(f"Wrote {len(runs)} runs to {csv_path}")


if __name__ == "__main__":
    main()
