#!/usr/bin/env python3
"""Summarize a scalar metric across runs from a W&B-exported CSV.

Expected format:
- one `Step` column
- many columns like:
  `<run_name> - <metric_name>`
- optional `__MIN` / `__MAX` companion columns, which are ignored

By default, the script takes the last non-empty value in each run's main metric
column. If `--steps` is provided, it instead aggregates the values at those
exact step rows.

Runs are grouped by method name by stripping the trailing `_seed<k>_job<id>`
suffix and reporting mean/std/min/max across seeds.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


RUN_SUFFIX_RE = re.compile(r"_seed\d+_job\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default="accuracy.csv")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="Exact Step values to summarize instead of final values, e.g. --steps 100 120 150",
    )
    return parser.parse_args()


def to_float(value: str) -> float | None:
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def method_name_from_run(run_name: str) -> str:
    return RUN_SUFFIX_RE.sub("", run_name)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        run_columns = []
        for name in fieldnames:
            if name == "Step":
                continue
            if name.endswith("__MIN") or name.endswith("__MAX"):
                continue
            if " - " not in name:
                continue
            run_columns.append(name)

        if not run_columns:
            raise SystemExit(f"No metric columns found in {csv_path}")

        metric_suffixes = sorted({col.split(" - ", 1)[1] for col in run_columns})
        if len(metric_suffixes) != 1:
            raise SystemExit(
                f"Expected exactly one metric suffix in {csv_path}, found: {metric_suffixes}"
            )
        run_col_suffix = f" - {metric_suffixes[0]}"

        step_targets = set(args.steps or [])
        step_values: dict[int, dict[str, float]] = {step: {} for step in step_targets}
        final_values: dict[str, float] = {}
        for row in reader:
            step_value = row.get("Step", "")
            step = None
            if step_value is not None and step_value.strip() != "":
                try:
                    step = int(float(step_value))
                except ValueError:
                    step = None
            for col in run_columns:
                value = to_float(row.get(col, ""))
                if value is not None:
                    final_values[col] = value
                    if step is not None and step in step_targets:
                        step_values[step][col] = value

    def print_grouped(title: str, values_by_col: dict[str, float]) -> None:
        grouped: dict[str, list[float]] = {}
        for col, value in values_by_col.items():
            run_name = col[: -len(run_col_suffix)]
            method = method_name_from_run(run_name)
            grouped.setdefault(method, []).append(value)

        if not grouped:
            raise SystemExit(f"No run columns matching `{run_col_suffix}` found in {csv_path}")

        print(title)
        for method in sorted(grouped):
            values = grouped[method]
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            vmin = min(values)
            vmax = max(values)
            print(f"{method} (n={len(values)}): {mean:.6f} +- {std:.6f}, min={vmin:.6f}, max={vmax:.6f}")
        print()

    print(f"CSV: {csv_path}")
    print(f"metric: {metric_suffixes[0]}")
    print()
    if args.steps:
        for step in args.steps:
            print_grouped(f"step={step}", step_values.get(step, {}))
    else:
        print_grouped("final", final_values)


if __name__ == "__main__":
    main()
