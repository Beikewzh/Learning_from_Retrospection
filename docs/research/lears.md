# LeaRS Online Training Reference

This document is the source-of-truth for the current LeaRS implementation in this repo.

> Update: for the latest async AR + token-level latent GRPO shaping details, see
> `docs/research/lears_async_token_grpo.md`.

## 1. What is implemented

LeaRS is integrated directly into the PPO loop (online-only behavior):

1. RL rollout produces responses.
2. Response-token latents are captured.
3. Latents are appended to `research/buffer`.
4. AR training is attempted in-loop on a deterministic cadence.
5. Intrinsic reward is computed from AR prediction error and added to token-level scores.

There is no external Phase A/B/C process restart path in the canonical scripts anymore.

LLM optimization path is unchanged from EasyR1 PPO. In the default `examples/research/lears_gsm8k.yaml`,
the actor uses LoRA adapters (`worker.actor.model.lora.rank > 0`), so PPO updates adapter weights.

## 2. Runtime behavior (single process, no re-init)

Within one `verl.trainer.main` run:

1. Warmup gate: AR training is blocked while `global_step < research.ar.start_after_steps`.
2. Cadence gate: AR training is only attempted when
   `(global_step - start_after_steps) % train_every_n_steps == 0`.
3. AR fit uses the latest buffer window (or full buffer if windowing disabled).
4. Intrinsic stays inactive until a scorer checkpoint is successfully loaded.
5. After scorer load, intrinsic is applied every step.

What this means for framework processes:

- EasyR1 trainer stays alive.
- vLLM stays in the same trainer run.
- AR updates happen inside that same run.
- No external offline AR subprocess is used by the canonical training scripts.

## 3. Data budget and AR update formulas

Definitions:

- `samples_per_step = ROLLOUT_BATCH_SIZE * ROLLOUT_N`
- `warmup_sample_budget = WARMUP_STEPS * samples_per_step`
- `start_after_steps = WARMUP_STEPS`
- `train_every_n_steps = ONLINE_AR_TRAIN_EVERY_N_STEPS` (full script) or `AR_TRAIN_EVERY_N_STEPS` (sweep/smoke)

AR attempt schedule:

- first possible attempt: `global_step = start_after_steps`
- next attempts: `start_after_steps + k * train_every_n_steps`

Expected AR attempt count over a run:

- `0`, if `TOTAL_STEPS < start_after_steps`
- `floor((TOTAL_STEPS - start_after_steps) / train_every_n_steps) + 1`, otherwise

Expected successful update count is `<=` attempt count, because attempts can be skipped for insufficient samples.

Warmup sufficiency auto-guard (full script):

- `min_warmup_steps_needed = ceil(AR_MIN_BUFFER_SAMPLES / samples_per_step)`
- effective warmup becomes `max(requested_warmup, min_warmup_steps_needed)`

## 4. Script entrypoints

Canonical scripts:

- full training: `scripts/slurm/lears_full_train.sbatch`
- 5-run sweep: `scripts/slurm/lears_sweep.sbatch`
- interactive smoke: `scripts/lears_smoke_1gpu.sh`

Legacy aliases (kept for compatibility):

- `scripts/slurm/lears_gsm8k_full_train.sbatch`
- `scripts/slurm/lears_gsm8k_sweep.sbatch`

## 5. Topology support

`scripts/slurm/lears_full_train.sbatch` supports:

- `1x1` (`trainer.nnodes=1`, `trainer.n_gpus_per_node=1`)
- `1x8` (`trainer.nnodes=1`, `trainer.n_gpus_per_node=8`)
- `2x4` (`trainer.nnodes=2`, `trainer.n_gpus_per_node=4`)

Preflight checks fail fast if requested trainer topology does not match the allocated Slurm topology.

For multi-node runs, optional `RAY_HEAD_IP` override is supported. If unset, head IP auto-detection is used.

## 6. Curriculum defaults

`CURRICULUM_STAGE` presets in full script:

- `stage0_math500` (method sanity)
- `stage1_main` (real training)

Current stage defaults in script:

| Stage | TRAIN_FILES | VAL_FILES | Intent |
|---|---|---|---|
| `stage0_math500` | `HuggingFaceH4/MATH-500@test` | `HuggingFaceH4/MATH-500@test` | short method validation |
| `stage1_main` | `hiyouga/math12k@train` | `hiyouga/math12k@test` | actual training |

If you need a train split for stage0, override `TRAIN_FILES` explicitly.

## 7. Variant presets

| Variant | `lambda_success` | `lambda_failure` | `AR_LR` | `AR_TRAIN_STEPS` fallback |
|---|---:|---:|---:|---:|
| `lears_default` | 0.1 | 0.1 | 1e-4 | 120 |
| `lears_weak` | 0.05 | 0.05 | 1e-4 | 120 |
| `lears_asym` | 0.05 | 0.2 | 1e-4 | 120 |
| `lears_fastar` | 0.1 | 0.1 | 3e-4 | 200 |

Stage presets set `AR_TRAIN_STEPS` first (`60` for stage0, `120` for stage1). If you want strict variant step counts, override `AR_TRAIN_STEPS` explicitly.

## 8. Full script runtime interface

Override precedence:

1. script defaults,
2. exported env vars,
3. inline submit-time overrides (`VAR=... sbatch ...`).

### 8.1 Core run controls

| Env | Default | Meaning |
|---|---|---|
| `WANDB_PROJECT_NAME` | `COMP767` | W&B project |
| `WANDB_ENTITY` | `zihan-wang-beike-mcgill-university` | W&B entity |
| `DRY_RUN` | `0` | print resolved plan and exit |
| `REPO_ROOT` | auto | repo root override |
| `USE_SINGULARITY` | `1` | run via singularity/apptainer |
| `SIF_PATH` | `${REPO_ROOT}/easyr1.sif` | container image |
| `BASE_CONFIG` | `examples/research/lears_gsm8k.yaml` | Hydra base config |
| `CACHE_ROOT` | `${REPO_ROOT}/.cache` | cache root |
| `SAVE_ROOT` | `${REPO_ROOT}/checkpoints/${WANDB_PROJECT_NAME}/lears_full` | run root |
| `RUN_ID` | auto | run id / resume key |
| `VARIANT` | `lears_default` | variant preset |
| `CURRICULUM_STAGE` | `stage0_math500` | data/profile preset |

### 8.2 Scheduler controls

| Env | Default | Meaning |
|---|---|---|
| `SCHEDULE_MODE` | `epoch` | `epoch` or `step` |
| `TOTAL_EPOCHS` | stage preset | total epochs in epoch mode |
| `WARMUP_EPOCHS` | stage preset | warmup epochs in epoch mode |
| `REFRESH_EPOCHS` | stage preset | cadence epochs in epoch mode |
| `TOTAL_STEPS` | `1868` (step mode fallback) | total RL steps |
| `WARMUP_STEPS` | `256` (step mode fallback) | warmup gate |
| `REFRESH_INTERVAL` | `256` (step mode fallback) | cadence base interval |
| `ONLINE_AR_TRAIN_EVERY_N_STEPS` | `${REFRESH_INTERVAL}` | in-loop AR cadence |

### 8.3 Data/rollout controls

| Env | Default | Meaning |
|---|---|---|
| `TRAIN_FILES` | stage preset | training dataset string |
| `VAL_FILES` | stage preset | validation dataset string |
| `PROMPT_KEY` | stage preset | prompt field |
| `ANSWER_KEY` | stage preset | answer field |
| `FORMAT_PROMPT` | stage preset | prompt template path |
| `MAX_PROMPT_LENGTH` | `512` | max prompt tokens |
| `MAX_RESPONSE_LENGTH` | `4096` | max response tokens |
| `ROLLOUT_BATCH_SIZE` | `8` | prompts per RL step |
| `ROLLOUT_N` | `2` | responses per prompt |
| `ACTOR_GLOBAL_BATCH_SIZE` | `8` | actor PPO global batch |
| `ROLLOUT_TP_SIZE` | auto (`1` on 1-GPU, else `2`) | vLLM tensor parallel |
| `SAVE_FREQ` | `50` | checkpoint frequency |

`ROLLOUT_N` must stay `> 1` for GRPO/RLOO.

### 8.4 AR + intrinsic controls

| Env | Default | Meaning |
|---|---|---|
| `AR_DEVICE` | `cpu` | AR train/score device (`cpu` or `cuda`) |
| `AR_D_MODEL` | `256` | AR hidden size |
| `AR_N_LAYERS` | `2` | AR layer count |
| `AR_N_HEADS` | `4` | AR head count |
| `AR_DROPOUT` | `0.1` | AR dropout |
| `AR_LR` | variant preset | AR learning rate |
| `AR_BATCH_SIZE` | `8` | AR batch size |
| `AR_TRAIN_STEPS` | stage/variant preset | optimizer steps per attempt |
| `AR_MIN_BUFFER_SAMPLES` | stage preset | minimum samples to fit AR |
| `AR_MAX_SAMPLES` | `100000` | max sampled sequences per attempt |
| `AR_WINDOW_INTERVALS` | `2` | number of recent intervals for AR data window |
| `AR_WINDOW_INTERVAL_STEPS` | `${REFRESH_INTERVAL}` | step width of each interval |
| `AR_MAX_AGE_STEPS` | `${REFRESH_INTERVAL}` | stale scorer threshold |
| `AR_STALE_ACTION` | `warn` | stale policy: `warn` or `fail` |
| `LAMBDA_SUCCESS` | variant preset | intrinsic weight for successful trajectories |
| `LAMBDA_FAILURE` | variant preset | intrinsic weight for failed trajectories |

Hydra mapping used by full script:

- `research.ar.start_after_steps <- WARMUP_STEPS`
- `research.ar.train_every_n_steps <- ONLINE_AR_TRAIN_EVERY_N_STEPS`

### 8.5 Topology and Ray controls

| Env | Default | Meaning |
|---|---|---|
| `TRAINER_NNODES` | allocation | trainer nodes (`1` or `2`) |
| `TRAINER_N_GPUS_PER_NODE` | inferred from allocation | trainer GPUs/node (`1`, `8`, or `4` in 2-node mode) |
| `RAY_HEAD_IP` | auto | optional Ray head IP override |
| `RAY_PORT` | `6379` | Ray GCS port |
| `RAY_DASHBOARD_PORT` | `8265` | dashboard port |
| `RAY_NUM_CPUS` | `${SLURM_CPUS_PER_TASK}` or `16` | Ray CPU advertized resources |

Trainer also honors external `RAY_ADDRESS` in `verl/trainer/main.py` for attaching to an existing Ray cluster.

### 8.6 Sweep script controls (`scripts/slurm/lears_sweep.sbatch`)

Sweep env knobs:

| Env | Default | Meaning |
|---|---|---|
| `MAX_STEPS` | `24` | run length per variant |
| `MAX_PROMPT_LENGTH` | `512` | prompt cap |
| `MAX_RESPONSE_LENGTH` | `4096` | response cap |
| `ROLLOUT_BATCH_SIZE` | `4` | prompts per step |
| `ACTOR_GLOBAL_BATCH_SIZE` | `4` | actor batch size |
| `ROLLOUT_N` | `2` | responses/prompt |
| `ROLLOUT_TP_SIZE` | `1` | vLLM TP size |
| `WARMUP_STEPS` | `8` | warmup gate |
| `AR_TRAIN_EVERY_N_STEPS` | `8` | in-loop AR cadence |
| `AR_DEVICE` | `cpu` | AR device |
| `AR_D_MODEL` | `256` | AR hidden size |
| `AR_N_LAYERS` | `2` | AR layers |
| `AR_N_HEADS` | `4` | AR heads |
| `AR_DROPOUT` | `0.1` | AR dropout |
| `AR_BATCH_SIZE` | `4` | AR batch size |
| `AR_MIN_BUFFER_SAMPLES` | `64` | min samples for AR fit |
| `AR_MAX_AGE_STEPS` | `${AR_TRAIN_EVERY_N_STEPS}` | stale threshold |
| `AR_STALE_ACTION` | `warn` | stale action |

## 9. W&B logging model

One run timeline per training job.

All LeaRS metrics are logged in the same W&B run as RL metrics.

Core LeaRS metrics:

- `research/intrinsic/active`
- `research/ar/attempted`
- `research/ar/trained`
- `research/ar/skipped`
- `research/ar/skip_warmup`
- `research/ar/skip_cadence`
- `research/ar/skip_insufficient_samples`
- `research/ar/windowed_samples`
- `research/ar/age_steps`
- `research/ar/stale`

## 10. Quick commands

1-GPU dry run of full script:

```bash
DRY_RUN=1 TRAINER_NNODES=1 TRAINER_N_GPUS_PER_NODE=1 \
SCHEDULE_MODE=step TOTAL_STEPS=4 WARMUP_STEPS=1 REFRESH_INTERVAL=1 \
sbatch --nodes=1 --gres=gpu:a100l:1 --cpus-per-task=16 --mem=120G --time=0-01:00 \
  scripts/slurm/lears_full_train.sbatch
```

1-GPU interactive smoke (single process, online AR):

```bash
ENABLE_WANDB=1 \
TOTAL_STEPS=4 WARMUP_STEPS=1 AR_TRAIN_EVERY_N_STEPS=1 \
MAX_PROMPT_LENGTH=128 MAX_RESPONSE_LENGTH=128 \
ROLLOUT_BATCH_SIZE=1 ACTOR_GLOBAL_BATCH_SIZE=1 ROLLOUT_N=2 \
AR_TRAIN_STEPS=1 AR_MIN_BUFFER_SAMPLES=1 AR_BATCH_SIZE=1 \
bash scripts/lears_smoke_1gpu.sh
```

1x8 full run:

```bash
CURRICULUM_STAGE=stage1_main \
SCHEDULE_MODE=epoch TOTAL_EPOCHS=2.0 WARMUP_EPOCHS=0.25 REFRESH_EPOCHS=0.25 \
VARIANT=lears_default AR_DEVICE=cuda \
sbatch scripts/slurm/lears_full_train.sbatch
```

2x4 full run:

```bash
CURRICULUM_STAGE=stage1_main \
TRAINER_NNODES=2 TRAINER_N_GPUS_PER_NODE=4 \
SCHEDULE_MODE=epoch TOTAL_EPOCHS=1.0 WARMUP_EPOCHS=0.25 REFRESH_EPOCHS=0.25 \
sbatch --nodes=2 --gres=gpu:a100l:4 scripts/slurm/lears_full_train.sbatch
```

5-run sweep:

```bash
sbatch scripts/slurm/lears_sweep.sbatch
```

## 11. Offline AR CLI (optional utility)

`scripts/slurm` does not call this in the online path. It remains for debugging/manual experiments:

- `research/scripts/train_ar_offline.py`

CLI args:

| Arg | Default | Meaning |
|---|---|---|
| `--buffer-dir` | required | latent buffer path |
| `--output-dir` | required | AR output/checkpoint dir |
| `--max-samples` | `100000` | max sampled sequences |
| `--global-step` | `0` | step tag / window upper bound |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--train-steps` | `200` | optimizer steps |
| `--batch-size` | `16` | mini-batch size |
| `--lr` | `1e-4` | learning rate |
| `--d-model` | `256` | AR hidden size |
| `--n-layers` | `4` | AR layer count |
| `--n-heads` | `8` | AR head count |
| `--dropout` | `0.1` | dropout |
| `--max-seq-len` | `2048` | max latent sequence length |
| `--min-buffer-samples` | `128` | min data threshold |
| `--window-intervals` | `0` | window count (`0` = full history) |
| `--interval-steps` | `0` | interval width (required if window enabled) |
| `--wandb-project` | empty | optional W&B project |
| `--wandb-entity` | empty | optional W&B entity |
| `--wandb-run-name` | empty | optional run name |
| `--wandb-group` | empty | optional run group |
| `--wandb-job-type` | `offline_ar` | optional job type |
| `--wandb-mode` | empty | optional mode |

## 12. Code map (what was added/changed)

- `research/config.py`
  - `research.ar.start_after_steps`
  - stale policy validation (`max_age_steps`, `stale_action`)
  - `latent.include_prompt=false` guard
- `research/manager/lears_manager.py`
  - warmup gate and deterministic cadence
  - AR lifecycle metrics (`attempted/trained/skipped/...`)
  - intrinsic activation only after scorer exists
  - AR freshness metric (`research/ar/age_steps`)
- `research/ar/scorer.py`
  - finite handling for short/no-transition rows
- `research/intrinsic/reward_rule.py`
  - `intrinsic_mask` interface (token-0 exclusion)
- `scripts/slurm/lears_full_train.sbatch`
  - single-process online orchestration for `1x1`, `1x8`, `2x4`
- `scripts/slurm/lears_sweep.sbatch`
  - baseline + online LeaRS variants
- `scripts/lears_smoke_1gpu.sh`
  - quick online smoke loop
- `tests/research/*`
  - scorer finite behavior, intrinsic masking, manager cadence/warmup/stale checks, config validation
