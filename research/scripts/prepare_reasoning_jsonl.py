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

"""Normalize a reasoning dataset into the repo's simple offline JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Source JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Normalized JSONL output path.")
    parser.add_argument("--id-key", type=str, default="id", help="Input key for example id.")
    parser.add_argument("--question-key", type=str, default="question", help="Input key for question text.")
    parser.add_argument("--answer-key", type=str, default="answer", help="Input key for answer text.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src, start=1):
            if args.limit is not None and written >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if args.question_key not in row or args.answer_key not in row:
                raise KeyError(
                    f"Line {idx} is missing required keys. "
                    f"Found keys={sorted(row.keys())}, expected question={args.question_key}, answer={args.answer_key}."
                )
            normalized = {
                "id": str(row.get(args.id_key, written)),
                "question": str(row[args.question_key]),
                "answer": str(row[args.answer_key]),
            }
            extras = {k: v for k, v in row.items() if k not in {args.id_key, args.question_key, args.answer_key}}
            if extras:
                normalized["meta"] = extras
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} normalized rows to {output_path}")


if __name__ == "__main__":
    main()
