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

"""Export HuggingFaceH4/MATH-500 to the simple local JSONL schema used by offline collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_mounted_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    p_str = str(p)
    if p.exists():
        return p.resolve()

    host_repo = os.environ.get("REPO_ROOT")
    if host_repo:
        host_repo_raw = str(Path(host_repo).expanduser())
        try:
            host_repo_path = Path(host_repo).expanduser().resolve()
            host_repo_str = str(host_repo_path)
        except Exception:
            host_repo_str = ""
        for prefix in [host_repo_raw, host_repo_str]:
            if prefix and (p_str == prefix or p_str.startswith(prefix + "/")):
                rel_str = p_str[len(prefix):].lstrip("/")
                return PROJECT_ROOT / rel_str

    if p_str.startswith("/workspace/") or p_str == "/workspace":
        return p
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/data/math500_local.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to export. Default: test",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to export.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Optional cache root. If set, configures HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE there.",
    )
    return parser.parse_args()


def configure_cache(cache_root: str | None) -> None:
    if cache_root is None:
        return
    root = resolve_repo_mounted_path(cache_root)
    hf_home = root / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["XDG_CACHE_HOME"] = str(root)
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_root)
    output_path = resolve_repo_mounted_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Cache configuration:")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME"]:
        print(f"  {key}={os.environ.get(key, '<unset>')}")

    dataset = load_dataset("HuggingFaceH4/MATH-500", split=args.split)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            if args.limit is not None and written >= args.limit:
                break
            payload = {
                "id": str(row.get("unique_id", i)),
                "question": row["problem"],
                "answer": row["answer"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
