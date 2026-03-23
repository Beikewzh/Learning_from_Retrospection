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

"""Pre-download a Hugging Face model/tokenizer and optionally a dataset into local cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional HF dataset id.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional HF dataset config.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Optional HF dataset split.")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Optional cache root. If unset, uses current HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE env vars.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
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
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_root)

    print("Cache configuration:")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"]:
        print(f"  {key}={os.environ.get(key, '<unset>')}")

    print(f"Downloading tokenizer for {args.model}")
    AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)

    print(f"Downloading model weights for {args.model}")
    AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        print(f"Downloading dataset {args.dataset} config={args.dataset_config} split={args.dataset_split}")
        load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)

    print("Done.")


if __name__ == "__main__":
    main()
