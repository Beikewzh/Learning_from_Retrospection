#!/bin/bash
#SBATCH --job-name=lears-gsm8k-4096
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00
#SBATCH --array=0-4%5
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-COMP767}"
WANDB_ENTITY="${WANDB_ENTITY:-zihan-wang-beike-mcgill-university}"
export WANDB_PROJECT_NAME WANDB_ENTITY

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_CONFIG="examples/research/lears_gsm8k.yaml"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/checkpoints/${WANDB_PROJECT_NAME}/lears_gsm8k_4096}"
mkdir -p "${SAVE_ROOT}"

COMMON_ARGS=(
    "config=${BASE_CONFIG}"
    "data.max_prompt_length=512"
    "data.max_response_length=4096"
    "data.rollout_batch_size=4"
    "worker.actor.global_batch_size=4"
    "worker.rollout.n=2"
    "worker.actor.padding_free=false"
    "research.latent.include_prompt=false"
    "research.ar.device=cpu"
    "research.ar.max_seq_len=4096"
    "research.ar.d_model=256"
    "research.ar.n_layers=2"
    "research.ar.n_heads=4"
    "research.ar.dropout=0.1"
    "research.ar.batch_size=4"
    "research.ar.min_buffer_samples=64"
    "trainer.project_name=${WANDB_PROJECT_NAME}"
)

run_train() {
    "${PYTHON_BIN}" -m verl.trainer.main "$@"
}

TASK_ID="${SLURM_ARRAY_TASK_ID}"
STAMP="$(date +%Y%m%d_%H%M%S)"

case "${TASK_ID}" in
    0)
        VARIANT="baseline_no_lears"
        ;;
    1)
        VARIANT="lears_default"
        LAMBDA_SUCCESS="0.1"
        LAMBDA_FAILURE="0.1"
        AR_LR="1e-4"
        AR_TRAIN_STEPS="30"
        ;;
    2)
        VARIANT="lears_weak"
        LAMBDA_SUCCESS="0.05"
        LAMBDA_FAILURE="0.05"
        AR_LR="1e-4"
        AR_TRAIN_STEPS="30"
        ;;
    3)
        VARIANT="lears_asym"
        LAMBDA_SUCCESS="0.05"
        LAMBDA_FAILURE="0.2"
        AR_LR="1e-4"
        AR_TRAIN_STEPS="30"
        ;;
    4)
        VARIANT="lears_fastar"
        LAMBDA_SUCCESS="0.1"
        LAMBDA_FAILURE="0.1"
        AR_LR="3e-4"
        AR_TRAIN_STEPS="60"
        ;;
    *)
        echo "Unsupported SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
        exit 1
        ;;
esac

RUN_ID="${VARIANT}_j${SLURM_JOB_ID}_t${TASK_ID}_${STAMP}"
RUN_SAVE_DIR="${SAVE_ROOT}/${RUN_ID}"
mkdir -p "${RUN_SAVE_DIR}"

echo "Running variant: ${VARIANT}"
echo "Checkpoint dir: ${RUN_SAVE_DIR}"

if [[ "${TASK_ID}" == "0" ]]; then
    run_train \
        "${COMMON_ARGS[@]}" \
        "research.enabled=false" \
        "trainer.max_steps=24" \
        "trainer.experiment_name=${RUN_ID}" \
        "trainer.logger=[\"file\",\"wandb\"]" \
        "trainer.save_checkpoint_path=${RUN_SAVE_DIR}" \
        "trainer.find_last_checkpoint=true"
    exit 0
fi

# Phase A: warmup RL to collect latents without online AR training.
run_train \
    "${COMMON_ARGS[@]}" \
    "research.enabled=true" \
    "research.intrinsic.lambda_success=${LAMBDA_SUCCESS}" \
    "research.intrinsic.lambda_failure=${LAMBDA_FAILURE}" \
    "research.ar.train_every_n_steps=1000000" \
    "trainer.max_steps=8" \
    "trainer.experiment_name=${RUN_ID}_warmup" \
    "trainer.logger=[\"file\"]" \
    "trainer.save_checkpoint_path=${RUN_SAVE_DIR}" \
    "trainer.find_last_checkpoint=true"

# Phase B: train AR model offline on GPU.
"${PYTHON_BIN}" research/scripts/train_ar_offline.py \
    --buffer-dir "${RUN_SAVE_DIR}/research/buffer" \
    --output-dir "${RUN_SAVE_DIR}/research/ar" \
    --device cuda \
    --global-step 8 \
    --max-seq-len 4096 \
    --batch-size 4 \
    --min-buffer-samples 64 \
    --lr "${AR_LR}" \
    --train-steps "${AR_TRAIN_STEPS}"

# Phase C: resume RL and use offline AR checkpoint for intrinsic scoring.
run_train \
    "${COMMON_ARGS[@]}" \
    "research.enabled=true" \
    "research.intrinsic.lambda_success=${LAMBDA_SUCCESS}" \
    "research.intrinsic.lambda_failure=${LAMBDA_FAILURE}" \
    "research.ar.train_every_n_steps=1000000" \
    "trainer.max_steps=24" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.logger=[\"file\",\"wandb\"]" \
    "trainer.save_checkpoint_path=${RUN_SAVE_DIR}" \
    "trainer.find_last_checkpoint=true"
