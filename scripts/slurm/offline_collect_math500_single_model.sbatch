#!/bin/bash
#SBATCH --job-name=offline-math500-one
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

join_cmd() {
    local out=""
    local arg
    for arg in "$@"; do
        out+=" $(printf '%q' "$arg")"
    done
    printf '%s' "${out# }"
}

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
cd "${REPO_ROOT}"
mkdir -p slurm_logs

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
DATA_PATH="${DATA_PATH:-}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-${REPO_ROOT}/examples/format_prompt/math.jinja}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/offline_math500_temp1_k32}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_ID="${MODEL_ID:-}"
MODEL_TAG="${MODEL_TAG:-}"
if [[ -z "${MODEL_ID}" ]]; then
    echo "MODEL_ID must be set" >&2
    exit 1
fi

LIMIT="${LIMIT:-500}"
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
FLUSH_EVERY_N_SAMPLES="${FLUSH_EVERY_N_SAMPLES:-1}"

USE_SINGULARITY="${USE_SINGULARITY:-1}"
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
        echo "No singularity/apptainer found. Set USE_SINGULARITY=0 only if host python has all dependencies." >&2
        exit 1
    fi
    if [[ ! -f "${SIF_PATH}" ]]; then
        echo "Missing container image: ${SIF_PATH}" >&2
        exit 1
    fi
fi

RUNTIME="/workspace"

to_runtime_path() {
    local host_path
    host_path="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1")"
    if [[ "${host_path}" == "${REPO_ROOT}" ]]; then
        printf '%s' "${RUNTIME}"
        return
    fi
    if [[ "${host_path}" == "${REPO_ROOT}/"* ]]; then
        printf '%s' "${RUNTIME}${host_path#${REPO_ROOT}}"
        return
    fi
    printf '%s' "${host_path}"
}

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
    DATA_PATH="${REPO_ROOT}/data/math500_local_${MODEL_TAG}.jsonl"
fi

RUNTIME_CACHE_ROOT="$(to_runtime_path "${CACHE_ROOT}")"
RUNTIME_DATA_PATH="$(to_runtime_path "${DATA_PATH}")"
RUNTIME_PROMPT_TEMPLATE="$(to_runtime_path "${PROMPT_TEMPLATE}")"
RUNTIME_OUTPUT_BASE="$(to_runtime_path "${OUTPUT_BASE}")"

run_shell() {
    local cmd="$1"
    if [[ "${USE_SINGULARITY}" == "1" ]]; then
        "${CONTAINER_BIN}" exec --nv --cleanenv \
            --bind "${REPO_ROOT}:${RUNTIME}" \
            "${SIF_PATH}" \
            bash -lc "cd ${RUNTIME} && ${cmd}"
    else
        bash -lc "cd ${REPO_ROOT} && ${cmd}"
    fi
}

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

build_env_prefix() {
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
log "Cache root: ${CACHE_ROOT}"
log "Data path: ${DATA_PATH}"
log "Output base: ${OUTPUT_BASE}"
log "Temperature: ${TEMPERATURE}, samples/question: ${NUM_SAMPLES_PER_QUESTION}, limit: ${LIMIT}"
log "Resume: ${RESUME}, overwrite: ${OVERWRITE}, flush_every_n_samples: ${FLUSH_EVERY_N_SAMPLES}"

log "Exporting MATH-500 to local JSONL"
run_shell "$(build_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd \
    ${PYTHON_BIN} research/scripts/export_math500_jsonl.py \
    --output "${RUNTIME_DATA_PATH}" \
    --split test \
    --limit "${LIMIT}" \
    --cache-root "${RUNTIME_CACHE_ROOT}")"

log "Pre-downloading ${MODEL_ID}"
run_shell "$(build_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd \
    ${PYTHON_BIN} research/scripts/predownload_hf_assets.py \
    --model "${MODEL_ID}" \
    --dataset HuggingFaceH4/MATH-500 \
    --dataset-split test \
    --cache-root "${RUNTIME_CACHE_ROOT}")"

if [[ "${PREDOWNLOAD_ONLY}" == "1" ]]; then
    log "Predownload only requested; exiting."
    exit 0
fi

output_dir="${RUNTIME_OUTPUT_BASE}/${MODEL_TAG}"
collect_cmd=(
    ${PYTHON_BIN} -m research.scripts.collect_latents_offline
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
    --flush-every-n-samples "${FLUSH_EVERY_N_SAMPLES}"
    --seed "${SEED}"
    --limit "${LIMIT}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    collect_cmd+=(--overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
    collect_cmd+=(--resume)
fi

log "Collecting ${MODEL_ID} -> ${output_dir}"
run_shell "$(build_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd "${collect_cmd[@]}")" &
child_pid=$!
wait "${child_pid}"
child_pid=""

log "Done."
