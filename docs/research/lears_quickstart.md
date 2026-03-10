# LeaRS Quickstart (Online-Only)

This is the shortest path to run the current LeaRS pipeline.

## 1. What runs

- one `verl.trainer.main` per job
- AR updates happen in-loop (no external phase restart)
- W&B shows RL + AR + intrinsic on one run timeline

## 2. Scripts

- full: `scripts/slurm/lears_full_train.sbatch`
- sweep: `scripts/slurm/lears_sweep.sbatch`
- local smoke: `scripts/lears_smoke_1gpu.sh`

Legacy aliases still work:

- `scripts/slurm/lears_gsm8k_full_train.sbatch`
- `scripts/slurm/lears_gsm8k_sweep.sbatch`

## 3. Core formulas

- `samples_per_step = ROLLOUT_BATCH_SIZE * ROLLOUT_N`
- `warmup_sample_budget = WARMUP_STEPS * samples_per_step`
- first AR attempt at `global_step = WARMUP_STEPS`
- then every `AR_TRAIN_EVERY_N_STEPS` (smoke/sweep) or `ONLINE_AR_TRAIN_EVERY_N_STEPS` (full)

## 4. Fast 1-GPU interactive smoke

Run inside your singularity shell at `/workspace`:

```bash
ENABLE_WANDB=1 \
TOTAL_STEPS=4 WARMUP_STEPS=1 AR_TRAIN_EVERY_N_STEPS=1 \
MAX_PROMPT_LENGTH=128 MAX_RESPONSE_LENGTH=128 \
ROLLOUT_BATCH_SIZE=1 ACTOR_GLOBAL_BATCH_SIZE=1 ROLLOUT_N=2 \
AR_TRAIN_STEPS=1 AR_MIN_BUFFER_SAMPLES=1 AR_BATCH_SIZE=1 \
bash scripts/lears_smoke_1gpu.sh
```

Smoke dataset:

- `examples/research/smoke_math.jsonl`

## 5. Full run examples

1x8:

```bash
CURRICULUM_STAGE=stage1_main \
SCHEDULE_MODE=epoch TOTAL_EPOCHS=2.0 WARMUP_EPOCHS=0.25 REFRESH_EPOCHS=0.25 \
VARIANT=lears_default AR_DEVICE=cuda \
sbatch scripts/slurm/lears_full_train.sbatch
```

2x4:

```bash
CURRICULUM_STAGE=stage1_main TRAINER_NNODES=2 TRAINER_N_GPUS_PER_NODE=4 \
SCHEDULE_MODE=epoch TOTAL_EPOCHS=1.0 WARMUP_EPOCHS=0.25 REFRESH_EPOCHS=0.25 \
sbatch --nodes=2 --gres=gpu:a100l:4 scripts/slurm/lears_full_train.sbatch
```

Dry-run only (no training launch):

```bash
DRY_RUN=1 sbatch scripts/slurm/lears_full_train.sbatch
```

## 6. W&B defaults

```bash
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-COMP767}"
WANDB_ENTITY="${WANDB_ENTITY:-zihan-wang-beike-mcgill-university}"
```

## 7. Key knobs you will usually touch

LLM updates are LoRA-based by default (`worker.actor.model.lora.*` in base config).

- schedule: `SCHEDULE_MODE`, `TOTAL_EPOCHS`/`TOTAL_STEPS`, `WARMUP_EPOCHS`/`WARMUP_STEPS`
- data: `TRAIN_FILES`, `VAL_FILES`, `MAX_RESPONSE_LENGTH`
- rollout: `ROLLOUT_BATCH_SIZE`, `ROLLOUT_N`
- AR: `AR_DEVICE`, `AR_TRAIN_STEPS`, `AR_MIN_BUFFER_SAMPLES`, `ONLINE_AR_TRAIN_EVERY_N_STEPS`
- intrinsic: `LAMBDA_SUCCESS`, `LAMBDA_FAILURE`

For the complete interface table, use:

- `docs/research/lears.md`
