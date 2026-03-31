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

"""Pre-download a Hugging Face model/tokenizer and optionally a dataset into local cache.

Model weights are fetched with huggingface_hub.snapshot_download only (files on disk).
We do not load the full model with AutoModelForCausalLM, which can OOM on login nodes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from research.scripts.offline_utils import PROJECT_ROOT, configure_hf_cache, resolve_repo_mounted_path


def _repo_on_scratch(scratch: str, repo: Path) -> bool:
    try:
        sc = Path(scratch).resolve()
        rr = repo.resolve()
        return rr == sc or str(rr).startswith(str(sc) + os.sep)
    except OSError:
        return False


def infer_default_cache_root() -> str:
    """Prefer <repo>/.cache when the repo is under $SCRATCH; else a scratch tree if $HOME clone."""
    repo = PROJECT_ROOT.resolve()
    scratch = os.environ.get("SCRATCH")
    if scratch and _repo_on_scratch(scratch, repo):
        return str(repo / ".cache")
    if scratch:
        return str(Path(scratch) / "Learning_from_Retrospection" / ".cache")
    return str(repo / ".cache")


def infer_default_local_dir(model_id: str) -> str | None:
    """Full model snapshot next to the repo on scratch when possible (hub id only)."""
    if Path(model_id).is_dir():
        return None
    scratch = os.environ.get("SCRATCH")
    repo = PROJECT_ROOT.resolve()
    safe = model_id.replace("/", "__")
    if scratch and _repo_on_scratch(scratch, repo):
        return str(repo / "models" / safe)
    if scratch:
        return str(Path(scratch) / "Learning_from_Retrospection" / "models" / safe)
    return None



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
        help="HF cache root (HF_HOME=huggingface/ under this). Default: <repo>/.cache if repo is under $SCRATCH, "
        "else $SCRATCH/Learning_from_Retrospection/.cache when SCRATCH is set, else <repo>/.cache.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Directory for a full model snapshot (for offline MODEL_PATH). Default if SCRATCH is set: "
        "$SCRATCH/Learning_from_Retrospection/models/<repo_id with / -> __>.",
    )
    parser.add_argument(
        "--no-local-snapshot",
        action="store_true",
        help="Do not write a copy under $SCRATCH/.../models/ (use hub cache only).",
    )
    return parser.parse_args()


def configure_cache(cache_root: str | None) -> None:
    configure_hf_cache(cache_root)


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root if args.cache_root is not None else infer_default_cache_root()
    configure_cache(cache_root)

    print("Cache configuration:")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"]:
        print(f"  {key}={os.environ.get(key, '<unset>')}")

    model_ref = args.model
    tokenizer_path = model_ref
    local_dir = args.local_dir
    if local_dir is None and not args.no_local_snapshot:
        local_dir = infer_default_local_dir(model_ref)

    if Path(model_ref).is_dir():
        print(f"Skipping model download; local path exists: {model_ref}")
    else:
        print(f"Downloading model repo files for {model_ref} (disk only, low RAM)")
        dl_kwargs = {
            "repo_id": model_ref,
            "repo_type": "model",
            "resume_download": True,
        }
        if local_dir:
            root = resolve_repo_mounted_path(local_dir)
            root.mkdir(parents=True, exist_ok=True)
            dl_kwargs["local_dir"] = str(root)
            tokenizer_path = str(root)
            print(f"  Snapshot directory: {root}")
        snapshot_download(**dl_kwargs)

    print(f"Verifying tokenizer for {tokenizer_path}")
    AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        print(f"Downloading dataset {args.dataset} config={args.dataset_config} split={args.dataset_split}")
        if args.dataset_config is not None:
            load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
        else:
            load_dataset(args.dataset, split=args.dataset_split)

    print("Done.")


if __name__ == "__main__":
    main()
