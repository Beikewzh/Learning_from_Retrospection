from __future__ import annotations

import os
from pathlib import Path


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
            host_repo_str = str(Path(host_repo).expanduser().resolve())
        except Exception:
            host_repo_str = ""
        for prefix in (host_repo_raw, host_repo_str):
            if prefix and (p_str == prefix or p_str.startswith(prefix + "/")):
                rel_str = p_str[len(prefix):].lstrip("/")
                return PROJECT_ROOT / rel_str

    if p_str.startswith("/workspace/") or p_str == "/workspace":
        return p
    return p


def configure_hf_cache(cache_root: str | None) -> Path | None:
    if cache_root is None:
        return None
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
    return root
