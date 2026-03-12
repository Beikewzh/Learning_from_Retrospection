# EasyR1 + LeaRS on HPC (Mila + Compute Canada)

This repo is designed to run with the Docker image:

- `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`

On Mila, use Singularity. On Compute Canada (for example Tamia), use Apptainer.

## 1. Pull image to SIF

### Mila (Singularity)

```bash
module load singularity/3.7.1

export SINGULARITY_CACHEDIR=$SCRATCH/.singularity_cache
export SINGULARITY_TMPDIR=$SCRATCH/.singularity_tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

cd $SCRATCH/Learning_from_Retrospection
singularity pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

### Compute Canada (Apptainer)

```bash
module load apptainer

export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
export APPTAINER_TMPDIR=$SCRATCH/.apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

cd $SCRATCH/Learning_from_Retrospection
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

On Tamia, this is typically available under `~/links/scratch/Learning_from_Retrospection`.

## 2. Interactive shell

### Mila (Singularity)

```bash
singularity shell --nv --cleanenv \
  --bind $SCRATCH/Learning_from_Retrospection:/workspace \
  easyr1.sif
cd /workspace
```

### Compute Canada (Apptainer)

```bash
apptainer shell --nv --cleanenv \
  --bind $SCRATCH/Learning_from_Retrospection:/workspace \
  easyr1.sif
cd /workspace
```

## 3. Canonical LeaRS scripts

- full online training: `scripts/slurm/lears_full_train.sbatch`
- online sweep: `scripts/slurm/lears_sweep.sbatch`
- quick interactive smoke: `scripts/lears_smoke_1gpu.sh`

Legacy aliases still exist:

- `scripts/slurm/lears_gsm8k_full_train.sbatch`
- `scripts/slurm/lears_gsm8k_sweep.sbatch`

## 4. Mila examples

1x8 full run:

```bash
sbatch scripts/slurm/lears_full_train.sbatch
```

1x1 dry-run preflight:

```bash
DRY_RUN=1 TRAINER_NNODES=1 TRAINER_N_GPUS_PER_NODE=1 \
SCHEDULE_MODE=step TOTAL_STEPS=4 WARMUP_STEPS=1 REFRESH_INTERVAL=1 \
sbatch --nodes=1 --gres=gpu:a100l:1 --cpus-per-task=16 --mem=120G --time=0-01:00 \
  scripts/slurm/lears_full_train.sbatch
```

2x4 full run:

```bash
TRAINER_NNODES=2 TRAINER_N_GPUS_PER_NODE=4 \
sbatch --nodes=2 --gres=gpu:a100l:4 scripts/slurm/lears_full_train.sbatch
```

1-GPU interactive smoke (5-minute sanity path):

```bash
ENABLE_WANDB=1 TOTAL_STEPS=4 WARMUP_STEPS=1 AR_TRAIN_EVERY_N_STEPS=1 \
AR_TRAIN_STEPS=1 AR_MIN_BUFFER_SAMPLES=1 AR_BATCH_SIZE=1 \
ROLLOUT_BATCH_SIZE=1 ACTOR_GLOBAL_BATCH_SIZE=1 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=128 MAX_RESPONSE_LENGTH=128 \
bash scripts/lears_smoke_1gpu.sh
```

## 5. Compute Canada templates

Set site-specific account/partition/GPU flags explicitly.

1x8 template:

```bash
sbatch \
  --account=<cc_account> \
  --partition=<cc_partition> \
  --nodes=1 \
  --gres=gpu:<gpu_type>:8 \
  --cpus-per-task=<cpus> \
  --mem=<mem> \
  --time=<walltime> \
  scripts/slurm/lears_full_train.sbatch
```

2x4 template:

```bash
TRAINER_NNODES=2 TRAINER_N_GPUS_PER_NODE=4 \
sbatch \
  --account=<cc_account> \
  --partition=<cc_partition> \
  --nodes=2 \
  --gres=gpu:<gpu_type>:4 \
  --cpus-per-task=<cpus> \
  --mem=<mem> \
  --time=<walltime> \
  scripts/slurm/lears_full_train.sbatch
```

If your CC site prefers `--gpus-per-node` over `--gres`, swap the resource flag accordingly.

## 6. W&B defaults

Scripts use:

```bash
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-COMP767}"
WANDB_ENTITY="${WANDB_ENTITY:-zihan-wang-beike-mcgill-university}"
```

If runs appear empty, verify `WANDB_API_KEY` is available in the runtime.

## 7. Resume and logs

Resume an existing run by reusing `RUN_ID`:

```bash
RUN_ID=<existing_run_id> sbatch scripts/slurm/lears_full_train.sbatch
```

Slurm logs are written to:

- `slurm_logs/slurm-<jobname>-<jobid>.out`
- `slurm_logs/slurm-<jobname>-<arrayjob>_<task>.out`
