#!/bin/bash
#SBATCH --job-name=offline-game24-one
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH="${SCRIPT_DIR}/offline_common.sh"
if [[ ! -f "${COMMON_SH}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/game24/offline_common.sh"
fi
if [[ ! -f "${COMMON_SH}" ]]; then
    echo "Cannot find offline_common.sh" >&2
    echo "Checked: ${SCRIPT_DIR}/offline_common.sh and ${SLURM_SUBMIT_DIR:-<unset>}/scripts/slurm/game24/offline_common.sh" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${COMMON_SH}"
offline_init_basic

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
DATA_JSON_SOURCE="${DATA_JSON_SOURCE:-${REPO_ROOT}/data/game24_reasoning/all_24_game_results_shuffled.json}"
DATA_PATH="${DATA_PATH:-}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-${REPO_ROOT}/research/offline/game24/game24_prompt.jinja}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/offline_game24_temp1_k32}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_ID="${MODEL_ID:-}"
MODEL_TAG="${MODEL_TAG:-}"
if [[ -z "${MODEL_ID}" ]]; then
    echo "MODEL_ID must be set" >&2
    exit 1
fi

LIMIT="${LIMIT:-500}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SAMPLES_PER_QUESTION="${NUM_SAMPLES_PER_QUESTION:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
LATENT_DTYPE="${LATENT_DTYPE:-fp16}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
LAYER_INDEX="${LAYER_INDEX:--1}"
SEED="${SEED:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"
PREDOWNLOAD_ONLY="${PREDOWNLOAD_ONLY:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"
FLUSH_EVERY_N_SAMPLES="${FLUSH_EVERY_N_SAMPLES:-1}"

USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
OFFLINE_ENABLE_NV=1

offline_setup_container

sanitize_model_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value//-/_}"
    value="${value//./_}"
    printf '%s' "${value,,}"
}

if [[ -z "${MODEL_TAG}" ]]; then
    MODEL_TAG="$(sanitize_model_name "${MODEL_ID}")"
fi
if [[ -z "${DATA_PATH}" ]]; then
    if [[ -n "${LIMIT}" ]]; then
        DATA_PATH="${REPO_ROOT}/data/game24_reasoning/game24_reasoning_limit${LIMIT}.jsonl"
    else
        DATA_PATH="${REPO_ROOT}/data/game24_reasoning/game24_reasoning.jsonl"
    fi
fi

RUNTIME_CACHE_ROOT="$(to_runtime_path "${CACHE_ROOT}")"
RUNTIME_DATA_JSON_SOURCE="$(to_runtime_path "${DATA_JSON_SOURCE}")"
RUNTIME_DATA_PATH="$(to_runtime_path "${DATA_PATH}")"
RUNTIME_PROMPT_TEMPLATE="$(to_runtime_path "${PROMPT_TEMPLATE}")"
RUNTIME_OUTPUT_BASE="$(to_runtime_path "${OUTPUT_BASE}")"

child_pid=""

handle_preemption() {
    log "Received preemption signal."
    if [[ -n "${child_pid}" ]]; then
        kill -TERM "${child_pid}" >/dev/null 2>&1 || true
        wait "${child_pid}" || true
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        log "Requeueing job ${SLURM_JOB_ID}"
        scontrol requeue "${SLURM_JOB_ID}" || true
    fi
    exit 0
}

trap handle_preemption TERM USR1

build_local_env_prefix() {
    local cache_root="$1"
    printf '%s' "$(join_cmd \
        export HF_HOME="${cache_root}/huggingface" \
        HF_HUB_CACHE="${cache_root}/huggingface/hub" \
        HUGGINGFACE_HUB_CACHE="${cache_root}/huggingface/hub" \
        HF_DATASETS_CACHE="${cache_root}/huggingface/datasets" \
        TRANSFORMERS_CACHE="${cache_root}/huggingface/hub" \
        XDG_CACHE_HOME="${cache_root}" \
        TORCH_HOME="${cache_root}/torch")"
}

mkdir -p "${CACHE_ROOT}" "$(dirname "${DATA_PATH}")" "${OUTPUT_BASE}"

log "Repo root: ${REPO_ROOT}"
log "Model: ${MODEL_ID}"
log "Model tag: ${MODEL_TAG}"
log "Data json source: ${DATA_JSON_SOURCE}"
log "Data path: ${DATA_PATH}"
log "Output base: ${OUTPUT_BASE}"
log "Temperature: ${TEMPERATURE}, samples/question: ${NUM_SAMPLES_PER_QUESTION}, limit: ${LIMIT}"
log "Sharding: num_shards=${NUM_SHARDS}, shard_index=${SHARD_INDEX}"
log "Resume: ${RESUME}, overwrite: ${OVERWRITE}, flush_every_n_samples: ${FLUSH_EVERY_N_SAMPLES}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
    if [[ ! -f "${DATA_JSON_SOURCE}" ]]; then
        echo "DATA_JSON_SOURCE does not exist: ${DATA_JSON_SOURCE}" >&2
        exit 1
    fi
    log "Preparing Game24 normalized JSONL"
    run_shell "$(build_local_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd \
        ${PYTHON_BIN} -m research.offline.game24.prepare_game24_jsonl \
        --input-json "${RUNTIME_DATA_JSON_SOURCE}" \
        --output-jsonl "${RUNTIME_DATA_PATH}" \
        --limit "${LIMIT}")"
fi

if [[ "${SKIP_PREDOWNLOAD}" != "1" ]]; then
    log "Pre-downloading ${MODEL_ID}"
    run_shell "$(build_local_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd \
        ${PYTHON_BIN} -m research.scripts.predownload_hf_assets \
        --model "${MODEL_ID}" \
        --cache-root "${RUNTIME_CACHE_ROOT}")"
fi

if [[ "${PREDOWNLOAD_ONLY}" == "1" ]]; then
    log "Predownload/prepare only requested; exiting."
    exit 0
fi

output_dir="${RUNTIME_OUTPUT_BASE}/${MODEL_TAG}_limit${LIMIT}"
if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    output_dir="${output_dir}/shard_$(printf '%02d' "${SHARD_INDEX}")"
fi

collect_cmd=(
    ${PYTHON_BIN} -m research.offline.game24.collect_latents_game24_offline
    --model "${MODEL_ID}"
    --input "${RUNTIME_DATA_PATH}"
    --output-dir "${output_dir}"
    --cache-root "${RUNTIME_CACHE_ROOT}"
    --prompt-template "${RUNTIME_PROMPT_TEMPLATE}"
    --layer-index "${LAYER_INDEX}"
    --torch-dtype "${TORCH_DTYPE}"
    --latent-dtype "${LATENT_DTYPE}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-response-length "${MAX_RESPONSE_LENGTH}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --num-samples-per-question "${NUM_SAMPLES_PER_QUESTION}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${SHARD_INDEX}"
    --flush-every-n-samples "${FLUSH_EVERY_N_SAMPLES}"
    --seed "$((SEED + SHARD_INDEX * 1000))"
    --limit "${LIMIT}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    collect_cmd+=(--overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
    collect_cmd+=(--resume)
fi

log "Collecting ${MODEL_ID} -> ${output_dir}"
run_shell "$(build_local_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd "${collect_cmd[@]}")" &
child_pid=$!
wait "${child_pid}"
child_pid=""

log "Done."
