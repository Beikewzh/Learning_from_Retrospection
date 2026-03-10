#!/bin/bash
#SBATCH --job-name=lears-online-sweep
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00
#SBATCH --array=0-4%5
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out

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
mkdir -p "${REPO_ROOT}/slurm_logs"

# Work around known site epilog issue.
touch "${REPO_ROOT}/.no_report" || true
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    touch "${SLURM_SUBMIT_DIR}/.no_report" || true
fi

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

BASE_CONFIG="${BASE_CONFIG:-examples/research/lears_gsm8k.yaml}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/checkpoints/${WANDB_PROJECT_NAME}/lears_online_sweep_4096}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
mkdir -p "${SAVE_ROOT}" "${CACHE_ROOT}/huggingface/hub" "${CACHE_ROOT}/huggingface/datasets" "${CACHE_ROOT}/torch"
if [[ "${USE_SINGULARITY}" == "1" && "${SAVE_ROOT}" != "${REPO_ROOT}"* ]]; then
    echo "SAVE_ROOT must be inside REPO_ROOT when USE_SINGULARITY=1, got: ${SAVE_ROOT}" >&2
    exit 1
fi

to_runtime_path() {
    local path="$1"
    if [[ "${USE_SINGULARITY}" == "1" && "${path}" == "${REPO_ROOT}"* ]]; then
        printf '/workspace%s' "${path#${REPO_ROOT}}"
    else
        printf '%s' "${path}"
    fi
}

run_python() {
    if [[ "${USE_SINGULARITY}" == "1" ]]; then
        local env_args=(
            --env "WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME}"
            --env "WANDB_ENTITY=${WANDB_ENTITY}"
            --env "HF_HOME=/workspace/.cache/huggingface"
            --env "HF_HUB_CACHE=/workspace/.cache/huggingface/hub"
            --env "HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub"
            --env "HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets"
            --env "XDG_CACHE_HOME=/workspace/.cache"
            --env "TORCH_HOME=/workspace/.cache/torch"
            --env "TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub"
            --env "PYTHONPATH=/workspace"
            --env "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-}"
            --env "WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-}"
        )
        local pass_vars=(
            WANDB_API_KEY HF_TOKEN HF_ENDPOINT WANDB_MODE
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

MAX_STEPS="${MAX_STEPS:-24}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"

WARMUP_STEPS="${WARMUP_STEPS:-8}"
AR_TRAIN_EVERY_N_STEPS="${AR_TRAIN_EVERY_N_STEPS:-8}"
AR_DEVICE="${AR_DEVICE:-cpu}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-4}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-64}"
AR_MAX_AGE_STEPS="${AR_MAX_AGE_STEPS:-${AR_TRAIN_EVERY_N_STEPS}}"
AR_STALE_ACTION="${AR_STALE_ACTION:-warn}"

if (( ROLLOUT_N < 2 )); then
    echo "ROLLOUT_N must be > 1 for GRPO/RLOO." >&2
    exit 1
fi
if (( WARMUP_STEPS < 0 || AR_TRAIN_EVERY_N_STEPS <= 0 )); then
    echo "WARMUP_STEPS must be >= 0 and AR_TRAIN_EVERY_N_STEPS must be > 0." >&2
    exit 1
fi
if [[ "${AR_DEVICE}" != "cpu" && "${AR_DEVICE}" != "cuda" ]]; then
    echo "AR_DEVICE must be one of: cpu, cuda." >&2
    exit 1
fi

case "${TASK_ID}" in
    0)
        VARIANT="baseline_no_lears"
        RESEARCH_ENABLED="false"
        ;;
    1)
        VARIANT="lears_default"
        RESEARCH_ENABLED="true"
        LAMBDA_SUCCESS="0.1"
        LAMBDA_FAILURE="0.1"
        AR_LR="1e-4"
        AR_TRAIN_STEPS="30"
        ;;
    2)
        VARIANT="lears_weak"
        RESEARCH_ENABLED="true"
        LAMBDA_SUCCESS="0.05"
        LAMBDA_FAILURE="0.05"
        AR_LR="1e-4"
        AR_TRAIN_STEPS="30"
        ;;
    3)
        VARIANT="lears_asym"
        RESEARCH_ENABLED="true"
        LAMBDA_SUCCESS="0.05"
        LAMBDA_FAILURE="0.2"
        AR_LR="1e-4"
        AR_TRAIN_STEPS="30"
        ;;
    4)
        VARIANT="lears_fastar"
        RESEARCH_ENABLED="true"
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

RUN_SAVE_DIR_RUNTIME="$(to_runtime_path "${RUN_SAVE_DIR}")"
BASE_CONFIG_RUNTIME="$(to_runtime_path "${BASE_CONFIG}")"

SAMPLES_PER_STEP=$((ROLLOUT_BATCH_SIZE * ROLLOUT_N))
WARMUP_SAMPLE_BUDGET=$((WARMUP_STEPS * SAMPLES_PER_STEP))
EXPECTED_AR_ATTEMPTS=0
if (( MAX_STEPS >= WARMUP_STEPS )); then
    EXPECTED_AR_ATTEMPTS=$((((MAX_STEPS - WARMUP_STEPS) / AR_TRAIN_EVERY_N_STEPS) + 1))
fi

export WANDB_RUN_GROUP="sweep_j${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
export WANDB_JOB_TYPE="${VARIANT}"

echo "Running variant=${VARIANT}, research_enabled=${RESEARCH_ENABLED}"
echo "Checkpoint dir: ${RUN_SAVE_DIR}"
echo "W&B: project=${WANDB_PROJECT_NAME}, entity=${WANDB_ENTITY}, run_group=${WANDB_RUN_GROUP}"
echo "W&B auth: WANDB_API_KEY=$([[ -n "${WANDB_API_KEY:-}" ]] && echo set || echo unset), WANDB_MODE=${WANDB_MODE:-online}"
echo "Online cadence: warmup_steps=${WARMUP_STEPS}, train_every_n_steps=${AR_TRAIN_EVERY_N_STEPS}, expected_attempts=${EXPECTED_AR_ATTEMPTS}"
echo "Data budget: samples_per_step=${SAMPLES_PER_STEP}, warmup_sample_budget=${WARMUP_SAMPLE_BUDGET}"

run_python -m verl.trainer.main \
    "config=${BASE_CONFIG_RUNTIME}" \
    "trainer.max_steps=${MAX_STEPS}" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.logger=[\"file\",\"wandb\"]" \
    "trainer.project_name=${WANDB_PROJECT_NAME}" \
    "trainer.save_checkpoint_path=${RUN_SAVE_DIR_RUNTIME}" \
    "trainer.find_last_checkpoint=true" \
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}" \
    "worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}" \
    "worker.rollout.n=${ROLLOUT_N}" \
    "worker.rollout.tensor_parallel_size=${ROLLOUT_TP_SIZE}" \
    "worker.rollout.val_override_config.n=1" \
    "worker.rollout.disable_tqdm=true" \
    "worker.actor.padding_free=false" \
    "research.enabled=${RESEARCH_ENABLED}" \
    "research.latent.include_prompt=false" \
    "research.ar.device=${AR_DEVICE}" \
    "research.ar.max_seq_len=${MAX_RESPONSE_LENGTH}" \
    "research.ar.d_model=${AR_D_MODEL}" \
    "research.ar.n_layers=${AR_N_LAYERS}" \
    "research.ar.n_heads=${AR_N_HEADS}" \
    "research.ar.dropout=${AR_DROPOUT}" \
    "research.ar.batch_size=${AR_BATCH_SIZE}" \
    "research.ar.lr=${AR_LR:-1e-4}" \
    "research.ar.train_steps=${AR_TRAIN_STEPS:-30}" \
    "research.ar.start_after_steps=${WARMUP_STEPS}" \
    "research.ar.train_every_n_steps=${AR_TRAIN_EVERY_N_STEPS}" \
    "research.ar.min_buffer_samples=${AR_MIN_BUFFER_SAMPLES}" \
    "research.ar.max_age_steps=${AR_MAX_AGE_STEPS}" \
    "research.ar.stale_action=${AR_STALE_ACTION}" \
    "research.intrinsic.lambda_success=${LAMBDA_SUCCESS:-0.0}" \
    "research.intrinsic.lambda_failure=${LAMBDA_FAILURE:-0.0}"
