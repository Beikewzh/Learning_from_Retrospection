#!/usr/bin/env python3
"""Emit bash `export` lines so lears_full_train.sbatch env matches merged PPO+yaml config.

Used when LEARS_FROZEN_YAML=1: one YAML is the source of truth; the launcher skips
the large COMMON_ARGS Hydra dotlist (except config= and optional MODEL_PATH).

Requires REPO_ROOT and BASE_CONFIG in the environment. Uses TRAINER_N_GPUS_PER_NODE
for rollout tensor-parallel default (same rule as the bash launcher: 1 -> 1, else 2).
"""
from __future__ import annotations

import os
import shlex
import sys
from typing import Any


def emit(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        s = "true" if value else "false"
    else:
        s = str(value)
    print(f"export {name}={shlex.quote(s)}")


def main() -> int:
    repo = os.environ.get("REPO_ROOT", "").strip()
    base = os.environ.get("BASE_CONFIG", "").strip()
    if not repo or not base:
        print("echo 'sync_lears_env_from_yaml: need REPO_ROOT and BASE_CONFIG' >&2", file=sys.stderr)
        return 1
    path = base if os.path.isabs(base) else os.path.join(repo, base)
    if not os.path.isfile(path):
        print(f"echo 'sync_lears_env_from_yaml: file not found: {path}' >&2", file=sys.stderr)
        return 1

    try:
        import yaml  # type: ignore
    except ImportError:
        print("echo 'sync_lears_env_from_yaml: PyYAML required' >&2", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    launcher = raw.pop("_lears_launcher", None) or {}
    if not isinstance(launcher, dict):
        launcher = {}

    from omegaconf import OmegaConf

    from verl.trainer.config import PPOConfig

    default_cfg = OmegaConf.structured(PPOConfig())
    file_cfg = OmegaConf.create(raw)
    cfg = OmegaConf.merge(default_cfg, file_cfg)
    cfg_obj = OmegaConf.to_object(cfg)
    cfg_obj.deep_post_init()

    d = cfg_obj.data
    w = cfg_obj.worker
    t = cfg_obj.trainer
    r = cfg_obj.research
    ar = r.ar
    buf = r.buffer
    intr = r.intrinsic
    tr = r.tracing

    emit("TRAIN_FILES", d.train_files)
    emit("VAL_FILES", d.val_files)
    emit("PROMPT_KEY", d.prompt_key)
    emit("ANSWER_KEY", d.answer_key)
    emit("FORMAT_PROMPT", d.format_prompt)

    emit("MAX_PROMPT_LENGTH", d.max_prompt_length)
    emit("MAX_RESPONSE_LENGTH", d.max_response_length)
    emit("ROLLOUT_BATCH_SIZE", d.rollout_batch_size)
    emit("ACTOR_GLOBAL_BATCH_SIZE", w.actor.global_batch_size)
    emit("ROLLOUT_N", w.rollout.n)

    # Keep shell summary aligned with Hydra (esp. LEARS_FROZEN_YAML=1, where rollout TP comes from YAML).
    emit("ROLLOUT_TP_SIZE", w.rollout.tensor_parallel_size)

    emit("SAVE_FREQ", t.save_freq)
    emit("EVAL_FREQ", t.val_freq)

    emit("AR_D_MODEL", ar.d_model)
    emit("AR_N_LAYERS", ar.n_layers)
    emit("AR_N_HEADS", ar.n_heads)
    emit("AR_DROPOUT", ar.dropout)
    emit("AR_LR", ar.lr)
    emit("AR_TRAIN_STEPS", ar.train_steps)
    emit("AR_BATCH_SIZE", ar.batch_size)
    emit("AR_MIN_BUFFER_SAMPLES", ar.min_buffer_samples)
    emit("AR_MAX_SAMPLES", buf.max_train_samples)
    emit("AR_DEVICE", ar.device)
    emit("AR_ASYNC_ENABLED", ar.async_enabled)
    emit("AR_ASYNC_QUEUE_SIZE", ar.async_queue_size)
    emit("AR_RELOAD_EVERY_N_STEPS", ar.reload_every_n_steps)

    emit("ONLINE_AR_TRAIN_EVERY_N_STEPS", ar.train_every_n_steps)

    emit("AR_WINDOW_INTERVALS", ar.window_intervals)
    if ar.window_interval_steps is not None:
        emit("AR_WINDOW_INTERVAL_STEPS", ar.window_interval_steps)

    if ar.max_age_steps is not None:
        emit("AR_MAX_AGE_STEPS", ar.max_age_steps)
    emit("AR_STALE_ACTION", ar.stale_action)

    emit("LAMBDA_SUCCESS", intr.lambda_success)
    emit("LAMBDA_FAILURE", intr.lambda_failure)
    emit("INTRINSIC_ETA", intr.eta)
    emit("INTRINSIC_GATE_MODE", intr.gate_mode)
    emit("INTRINSIC_LAMBDA_SUCCESS_GATE", intr.lambda_success_gate)
    emit("INTRINSIC_SMOOTHING_WINDOW", intr.temporal_smoothing_window)
    emit("INTRINSIC_NORMALIZE_SCOPE", intr.normalize_scope)
    emit("INTRINSIC_GROUP_NORM_PER_TIMESTEP", intr.group_norm_per_timestep)
    emit("INTRINSIC_CLIP_VALUE", intr.clip_value)

    emit("BUFFER_FLUSH_EVERY_N_STEPS", buf.flush_every_n_steps)

    emit("TRACE_ENABLED", tr.enabled)
    emit("TRACE_FLUSH_EVERY_N_STEPS", tr.flush_every_n_steps)
    emit("TRACE_RETENTION_MODE", tr.retention_mode)
    emit("TRACE_MAX_DISK_GB", tr.max_disk_gb)
    emit("TRACE_SAMPLE_RATE", tr.sample_rate)
    emit("TRACE_SAVE_TOKENS", tr.save_tokens)
    emit("TRACE_SAVE_DECODED_TEXT", tr.save_decoded_text)
    emit("TRACE_SAVE_LATENTS", tr.save_latents)
    emit("TRACE_SAVE_AR_ERROR", tr.save_ar_error)

    mode = str(launcher.get("schedule_mode") or "epoch").strip()
    emit("SCHEDULE_MODE", mode)

    if mode == "epoch":
        te = launcher.get("total_epochs")
        if te is None:
            te = t.total_epochs
        emit("TOTAL_EPOCHS", te)
        emit("WARMUP_EPOCHS", launcher.get("warmup_epochs", 0.25))
        emit("REFRESH_EPOCHS", launcher.get("refresh_epochs", 0.25))
    else:
        emit("TOTAL_STEPS", launcher.get("total_steps", 1868))
        emit("WARMUP_STEPS", launcher.get("warmup_steps", 256))
        emit("REFRESH_INTERVAL", launcher.get("refresh_interval", 256))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
