#!/bin/bash
#SBATCH --job-name=lears-gsm8k-4096
#SBATCH --partition=long
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
USE_SINGULARITY="${USE_SINGULARITY:-1}"
if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
CONTAINER_BIN=""
if [[ "${USE_SINGULARITY}" == "1" ]]; then
    SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${HOME}/scratch/.singularity_cache}"
    SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${HOME}/scratch/.singularity_tmp}"
    export SINGULARITY_CACHEDIR SINGULARITY_TMPDIR
    mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_BIN="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_BIN="apptainer"
    else
        if [[ -f /etc/profile ]]; then
            # Some site profile scripts assume undefined vars unless nounset is disabled.
            set +u
            # shellcheck source=/dev/null
            source /etc/profile >/dev/null 2>&1 || true
            set -u
        fi
        if command -v module >/dev/null 2>&1; then
            module load singularity/3.7.1 >/dev/null 2>&1 || true
        fi
        if command -v singularity >/dev/null 2>&1; then
            CONTAINER_BIN="singularity"
        elif command -v apptainer >/dev/null 2>&1; then
            CONTAINER_BIN="apptainer"
        fi
    fi
    if [[ -z "${CONTAINER_BIN}" ]]; then
        echo "No singularity/apptainer found. Set USE_SINGULARITY=0 to run host python." >&2
        exit 1
    fi
    if [[ ! -f "${SIF_PATH}" ]]; then
        echo "Container image not found: ${SIF_PATH}" >&2
        exit 1
    fi
fi

BASE_CONFIG="examples/research/lears_gsm8k.yaml"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/checkpoints/${WANDB_PROJECT_NAME}/lears_gsm8k_4096}"
mkdir -p "${SAVE_ROOT}"

# Work around a known cluster epilog parsing issue for StdOut.
# If this marker exists, epilog skips its extra GPU-report redirection.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    touch "${SLURM_SUBMIT_DIR}/.no_report" || true
fi

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
    run_python -m verl.trainer.main "$@"
}

run_python() {
    if [[ "${USE_SINGULARITY}" == "1" ]]; then
        local env_args=(
            --env "WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME}"
            --env "WANDB_ENTITY=${WANDB_ENTITY}"
        )
        local pass_vars=(
            WANDB_API_KEY HF_TOKEN HF_HOME HF_ENDPOINT HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
            HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
        )
        local var
        for var in "${pass_vars[@]}"; do
            if [[ -n "${!var:-}" ]]; then
                env_args+=(--env "${var}=${!var}")
            fi
        done

        local cmd
        printf -v cmd '%q ' "${PYTHON_BIN}" "$@"
        "${CONTAINER_BIN}" exec \
            --nv \
            --cleanenv \
            --bind "${REPO_ROOT}:/workspace" \
            "${env_args[@]}" \
            "${SIF_PATH}" \
            bash -lc "cd /workspace && ${cmd}"
    else
        "${PYTHON_BIN}" "$@"
    fi
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
run_python research/scripts/train_ar_offline.py \
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
