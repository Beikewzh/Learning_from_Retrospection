# Async LeaRS + Token-Level Latent GRPO (Current Implementation)

This document is the current source-of-truth for the async LeaRS integration with token-level latent GRPO shaping in this repo.

## 1) What the training process is right now

Training runs in a single EasyR1 PPO job. There is no external phase restart.

Per PPO step, the flow is:

1. Generate rollouts with vLLM.
2. Compute reward (`token_level_scores`) as usual.
3. If `research.enabled=true`:
   1. Capture response latents.
   2. Append latents to buffer.
   3. Poll async AR futures (non-blocking).
   4. Optionally hot-reload scorer checkpoint (`reload_every_n_steps` cadence).
   5. Try to enqueue AR retraining on cadence (`train_every_n_steps`) after warmup.
   6. Score token-level AR prediction error online for the current batch.
   7. Apply intrinsic preprocessing (smoothing + normalization + clipping).
4. Compute external GRPO advantage/returns using existing EasyR1 path.
5. Compose total advantage for GRPO:
   - `A_total(i,t) = A_ext(i,t) + eta * g_i * A_int(i,t)`
6. Update actor/critic as normal.

Key code path:

- `verl/trainer/ray_trainer.py`:
  - manager creation guarded by `config.research.enabled`
  - per-step collect/score + advantage composition
- `research/manager/lears_manager.py`:
  - async AR queue/poll/reload
  - latent append, intrinsic scoring, stale guard

## 2) Where the modified GRPO method is

Main integration point:

- `verl/trainer/ray_trainer.py`
  - inside `fit()` -> `with timer("adv", ...)`
  - keeps original external reward + `compute_advantage(...)`
  - then adds intrinsic branch only when:
    - `research.enabled=true`
    - `adv_estimator in {grpo, grpo_passk}`
    - `research.intrinsic.eta > 0`

Formula implementation:

- gate computation: `research/intrinsic/reward_rule.py::compute_intrinsic_gate`
- composition helper: `research/intrinsic/reward_rule.py::compose_total_advantage`

External reward semantics are unchanged:

- reward function pipeline remains `token_level_scores`
- `token_level_rewards` and returns are computed by existing EasyR1 path
- only `advantages` are augmented for the actor objective when intrinsic is enabled

## 3) Async AR sidecar behavior

Implemented in `research/manager/lears_manager.py`:

- Non-blocking sidecar with `ThreadPoolExecutor(max_workers=1)` when `research.ar.async_enabled=true`.
- Step path:
  - append latent sample to buffer/recent reservoir
  - poll completed AR futures
  - reload scorer on step cadence
  - compute intrinsic for current batch
- Background path:
  - on cadence, enqueue AR training future
  - write checkpoint `latest.pt`
  - hot-swap scorer on step boundary via poll/reload
- Stale guard:
  - `max_age_steps` + `stale_action in {warn, fail}`

## 4) Intrinsic token preprocessing and gates

Implemented in `research/intrinsic/reward_rule.py`.

Token-level intrinsic `A_int` comes from AR prediction error with:

- causal temporal moving average (`temporal_smoothing_window`)
- normalization (`normalize_scope`):
  - `per_sequence_zscore`
  - `running_zscore`
  - `none`
- clipping (`clip_value`)
- masked first-token exclusion done in manager before preprocessing

Gate registry:

- `failure_only`: `g_i = 1 - R_ext(i)` (binary success thresholded by `success_threshold`)
- `asymmetric`: `g_i = 1` for failures, `g_i = -lambda_success_gate` for successes
- `none`: `g_i = 1`

## 5) Tracing and saved statistics

Async trace writer:

- `research/trace_writer.py` (`AsyncTraceWriter`)
- append-only JSONL shards + optional latent `.pt` blobs
- async queue + periodic flush to reduce hot-path overhead

Trace payload includes:

- run/sample identifiers: `run_id`, `step`, `uid`, `sample_idx`
- extrinsic outcome: `extrinsic_final`, `success`
- token ids (if enabled)
- AR error stats + token values (if enabled)
- intrinsic stats + token values
- gate/eta and intrinsic contribution (`eta * gate * intrinsic`)
- schema version is written per record (`schema_version`)

Retention modes:

- `keep_all`
- `rolling_budget`
- `sampled`

Note: tracing config has a `compression` field, but records are currently written as JSONL shards (not compressed-on-write).

## 6) Config surface

Core additions:

- `research.ar.async_enabled`
- `research.ar.async_queue_size`
- `research.ar.reload_every_n_steps`
- `research.intrinsic.eta`
- `research.intrinsic.gate_mode`
- `research.intrinsic.lambda_success_gate`
- `research.intrinsic.temporal_smoothing_window`
- `research.intrinsic.normalize_scope`
- `research.intrinsic.clip_value`
- `research.tracing.enabled`
- `research.tracing.save_tokens`
- `research.tracing.save_decoded_text`
- `research.tracing.save_latents`
- `research.tracing.save_ar_error`
- `research.tracing.retention_mode`
- `research.tracing.schema_version`
- `data.dataset_preset`

Dataset preset registry:

- `verl/trainer/dataset_presets.py`
- current preset: `math12k_math500`

## 7) Starting hyperparameters for Math12k -> Math500

Ready config:

- `examples/research/lears_math12k_math500.yaml`

Includes:

- `rollout.n=2`
- `max_prompt_length=512`
- `max_response_length=2048`
- actor lr `1e-5` (LoRA path)
- AR: `d_model=256`, `n_layers=2`, `n_heads=4`, `dropout=0.1`
- AR train: `lr=1e-4`, `batch_size=8`, `train_steps=60`, `train_every_n_steps=32`
- intrinsic: `eta=0.05`, `gate_mode=failure_only`, `temporal_smoothing_window=3`, `normalize_scope=per_sequence_zscore`, `clip_value=3.0`
- deterministic val override (`temperature=0`, `n=1`) and `val_freq=100`

Script defaults updated for this path:

- `scripts/slurm/lears_full_train.sbatch`

## 8) What changed vs original EasyR1

When `research.enabled=false`:

- `LeaRSManager` is not created.
- intrinsic scoring/composition is skipped.
- EasyR1 reward -> advantage -> update path remains original.

When enabled:

- changes are additive around the `adv` section in trainer.
- no rewrite of core actor loss function; only `advantages` tensor is augmented.

Primary non-research files touched:

- `verl/trainer/ray_trainer.py` (integration hooks only)
- `verl/trainer/config.py` (dataset preset field + post-init hook)
- `scripts/slurm/lears_full_train.sbatch` (runtime defaults/knobs)

## 9) Tests currently covering this integration

Passing research tests include:

- intrinsic smoothing/zscore/gating/composition
- manager cadence/warmup/stale/reload behavior
- dataset preset behavior
- config validation

Command used:

```bash
singularity exec --nv --cleanenv \
  --bind /network/scratch/z/zihan.wang/Learning_from_Retrospection:/workspace \
  --pwd /workspace easyr1.sif \
  python -m pytest -q tests/research
```

At the time of writing: all `tests/research` passed.

## 10) Remaining items if you want strict 100% acceptance coverage

- add explicit end-to-end integration smoke test that PPO step throughput continues during async AR retraining
- add benchmark report comparing sync vs async `timing_s/adv`
- optionally add compressed trace shards if required by your storage policy
