#!/bin/bash
set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-COMP767}"
WANDB_ENTITY="${WANDB_ENTITY:-pingsheng}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
export WANDB_PROJECT_NAME WANDB_ENTITY

PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"

if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
CONTAINER_BIN=""
if [[ "${USE_SINGULARITY}" == "1" ]]; then
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_BIN="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_BIN="apptainer"
    else
        # Common interactive path: this script is executed inside an existing container.
        if [[ -n "${SINGULARITY_NAME:-}" || -n "${APPTAINER_NAME:-}" || -d "/.singularity.d" ]]; then
            log "No singularity/apptainer binary inside container; falling back to USE_SINGULARITY=0."
            USE_SINGULARITY=0
        else
            echo "No singularity/apptainer found. Set USE_SINGULARITY=0 for host runtime." >&2
            exit 1
        fi
    fi
    if [[ "${USE_SINGULARITY}" == "1" && ! -f "${SIF_PATH}" ]]; then
        echo "Container image not found: ${SIF_PATH}" >&2
        exit 1
    fi
fi

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/checkpoints/${WANDB_PROJECT_NAME}/lears_smoke_1gpu}"
mkdir -p "${CACHE_ROOT}/huggingface/hub" "${CACHE_ROOT}/huggingface/datasets" "${CACHE_ROOT}/torch" "${SAVE_ROOT}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

BASE_CONFIG="${BASE_CONFIG:-examples/research/lears_gsm8k.yaml}"
TRAIN_FILES="${TRAIN_FILES:-${REPO_ROOT}/examples/research/smoke_math.jsonl}"
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
FORMAT_PROMPT="${FORMAT_PROMPT:-./examples/format_prompt/math.jinja}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

RUN_ID="${RUN_ID:-smoke1gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${SAVE_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

TOTAL_STEPS="${TOTAL_STEPS:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
AR_TRAIN_EVERY_N_STEPS="${AR_TRAIN_EVERY_N_STEPS:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-128}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
SAVE_FREQ="${SAVE_FREQ:-1}"

AR_DEVICE="${AR_DEVICE:-cpu}"
AR_D_MODEL="${AR_D_MODEL:-64}"
AR_N_LAYERS="${AR_N_LAYERS:-1}"
AR_N_HEADS="${AR_N_HEADS:-2}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_LR="${AR_LR:-1e-4}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-1}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-1}"
AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-1}"
LAMBDA_SUCCESS="${LAMBDA_SUCCESS:-0.1}"
LAMBDA_FAILURE="${LAMBDA_FAILURE:-0.1}"

if (( ROLLOUT_N < 2 )); then
    echo "ROLLOUT_N must be >= 2 for GRPO." >&2
    exit 1
fi
if (( TOTAL_STEPS <= 0 || WARMUP_STEPS < 0 || AR_TRAIN_EVERY_N_STEPS <= 0 )); then
    echo "TOTAL_STEPS must be > 0, WARMUP_STEPS >= 0, AR_TRAIN_EVERY_N_STEPS > 0." >&2
    exit 1
fi
if [[ "${AR_DEVICE}" != "cpu" && "${AR_DEVICE}" != "cuda" ]]; then
    echo "AR_DEVICE must be one of: cpu, cuda." >&2
    exit 1
fi

if [[ "${ENABLE_WANDB}" == "1" ]]; then
    LOGGER_VALUE='["file","wandb"]'
else
    LOGGER_VALUE='["file"]'
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
            --env "WANDB_RUN_GROUP=${RUN_ID}"
            --env "WANDB_JOB_TYPE=smoke_online"
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

BASE_CONFIG_RT="$(to_runtime_path "${BASE_CONFIG}")"
TRAIN_FILES_RT="$(to_runtime_path "${TRAIN_FILES}")"
VAL_FILES_RT="$(to_runtime_path "${VAL_FILES}")"
FORMAT_PROMPT_RT="$(to_runtime_path "${FORMAT_PROMPT}")"
RUN_DIR_RT="$(to_runtime_path "${RUN_DIR}")"

SAMPLES_PER_STEP=$((ROLLOUT_BATCH_SIZE * ROLLOUT_N))
WARMUP_SAMPLE_BUDGET=$((WARMUP_STEPS * SAMPLES_PER_STEP))
EXPECTED_AR_ATTEMPTS=0
if (( TOTAL_STEPS >= WARMUP_STEPS )); then
    EXPECTED_AR_ATTEMPTS=$((((TOTAL_STEPS - WARMUP_STEPS) / AR_TRAIN_EVERY_N_STEPS) + 1))
fi

log "Smoke run id: ${RUN_ID}"
log "Run directory: ${RUN_DIR}"
log "Dataset: train=${TRAIN_FILES} val=${VAL_FILES}"
log "Model: ${MODEL_PATH}"
log "Online schedule: total_steps=${TOTAL_STEPS}, warmup_steps=${WARMUP_STEPS}, train_every_n_steps=${AR_TRAIN_EVERY_N_STEPS}"
log "Data budget: samples_per_step=${SAMPLES_PER_STEP}, warmup_sample_budget=${WARMUP_SAMPLE_BUDGET}, expected_ar_attempts=${EXPECTED_AR_ATTEMPTS}"
if [[ "${ENABLE_WANDB}" == "1" ]]; then
    log "W&B logging enabled (project=${WANDB_PROJECT_NAME}, entity=${WANDB_ENTITY})."
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        log "WANDB_API_KEY is not exported. Ensure prior wandb login is available in this runtime."
    fi
else
    log "W&B logging disabled (ENABLE_WANDB=0)."
fi

run_python -m verl.trainer.main \
    "config=${BASE_CONFIG_RT}" \
    "data.train_files=${TRAIN_FILES_RT}" \
    "data.val_files=${VAL_FILES_RT}" \
    "data.prompt_key=question" \
    "data.answer_key=answer" \
    "data.format_prompt=${FORMAT_PROMPT_RT}" \
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}" \
    "data.val_batch_size=1" \
    "worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}" \
    "worker.actor.micro_batch_size_per_device_for_update=1" \
    "worker.actor.micro_batch_size_per_device_for_experience=1" \
    "worker.critic.micro_batch_size_per_device_for_update=1" \
    "worker.critic.micro_batch_size_per_device_for_experience=1" \
    "worker.rollout.n=${ROLLOUT_N}" \
    "worker.rollout.tensor_parallel_size=1" \
    "worker.rollout.val_override_config.n=1" \
    "worker.rollout.disable_tqdm=true" \
    "worker.actor.padding_free=false" \
    "worker.actor.model.model_path=${MODEL_PATH}" \
    "worker.actor.model.tokenizer_path=${MODEL_PATH}" \
    "trainer.nnodes=1" \
    "trainer.n_gpus_per_node=1" \
    "trainer.max_try_make_batch=5" \
    "trainer.val_freq=-1" \
    "trainer.val_before_train=false" \
    "trainer.val_generations_to_log=0" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.save_limit=2" \
    "trainer.project_name=${WANDB_PROJECT_NAME}" \
    "trainer.save_checkpoint_path=${RUN_DIR_RT}" \
    "trainer.find_last_checkpoint=true" \
    "trainer.logger=${LOGGER_VALUE}" \
    "trainer.max_steps=${TOTAL_STEPS}" \
    "trainer.experiment_name=${RUN_ID}_online" \
    "research.enabled=true" \
    "research.latent.include_prompt=false" \
    "research.ar.device=${AR_DEVICE}" \
    "research.ar.max_seq_len=${MAX_RESPONSE_LENGTH}" \
    "research.ar.d_model=${AR_D_MODEL}" \
    "research.ar.n_layers=${AR_N_LAYERS}" \
    "research.ar.n_heads=${AR_N_HEADS}" \
    "research.ar.dropout=${AR_DROPOUT}" \
    "research.ar.batch_size=${AR_BATCH_SIZE}" \
    "research.ar.lr=${AR_LR}" \
    "research.ar.train_steps=${AR_TRAIN_STEPS}" \
    "research.ar.min_buffer_samples=${AR_MIN_BUFFER_SAMPLES}" \
    "research.ar.start_after_steps=${WARMUP_STEPS}" \
    "research.ar.train_every_n_steps=${AR_TRAIN_EVERY_N_STEPS}" \
    "research.intrinsic.lambda_success=${LAMBDA_SUCCESS}" \
    "research.intrinsic.lambda_failure=${LAMBDA_FAILURE}"

log "Online smoke finished. Checkpoints at: ${RUN_DIR}"
