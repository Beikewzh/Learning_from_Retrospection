#!/bin/bash
#SBATCH --job-name=offline-ar-split
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
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

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "RUN_DIR must be set" >&2
    exit 1
fi
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
CONTAINER_BIN=""

METRICS_PATH="${METRICS_PATH:-${RUN_DIR}/spectrum_parallel/merged/metrics.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-${RUN_DIR}/ar_split_eval_parallel}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
SEED="${SEED:-1}"
USE_REASONING_SPAN="${USE_REASONING_SPAN:-1}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"
AR_DEVICE="${AR_DEVICE:-cpu}"
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-1000}"
AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-100}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-8}"
AR_LR="${AR_LR:-1e-4}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN:-4096}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-64}"
SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES:-1}"
KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${USE_SINGULARITY}" == "1" ]]; then
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_BIN="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_BIN="apptainer"
    else
        if [[ -f /etc/profile ]]; then
            set +u
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
        echo "No singularity/apptainer found" >&2
        exit 1
    fi
fi

RUNTIME="/workspace"

to_runtime_path() {
    local host_path
    local repo_root_resolved
    host_path="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1")"
    repo_root_resolved="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${REPO_ROOT}")"
    if [[ "${host_path}" == "${REPO_ROOT}" || "${host_path}" == "${repo_root_resolved}" ]]; then
        printf '%s' "${RUNTIME}"
        return
    fi
    if [[ "${host_path}" == "${REPO_ROOT}/"* ]]; then
        printf '%s' "${RUNTIME}${host_path#${REPO_ROOT}}"
        return
    fi
    if [[ "${host_path}" == "${repo_root_resolved}/"* ]]; then
        printf '%s' "${RUNTIME}${host_path#${repo_root_resolved}}"
        return
    fi
    printf '%s' "${host_path}"
}

build_env_prefix() {
    local cache_root="$1"
    printf '%s' "$(join_cmd \
        export REPO_ROOT="${REPO_ROOT}" \
        PROJECT_ROOT="${RUNTIME}" \
        HF_HOME="${cache_root}/huggingface" \
        HF_HUB_CACHE="${cache_root}/huggingface/hub" \
        HUGGINGFACE_HUB_CACHE="${cache_root}/huggingface/hub" \
        HF_DATASETS_CACHE="${cache_root}/huggingface/datasets" \
        TRANSFORMERS_CACHE="${cache_root}/huggingface/hub" \
        XDG_CACHE_HOME="${cache_root}" \
        TORCH_HOME="${cache_root}/torch")"
}

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

RUNTIME_CACHE_ROOT="$(to_runtime_path "${CACHE_ROOT}")"
RUNTIME_RUN_DIR="$(to_runtime_path "${RUN_DIR}")"
RUNTIME_METRICS_PATH="$(to_runtime_path "${METRICS_PATH}")"
SPLIT_OUTPUT_DIR="${OUTPUT_BASE}/split_$(printf '%02d' "${SPLIT_INDEX}")"
RUNTIME_SPLIT_OUTPUT_DIR="$(to_runtime_path "${SPLIT_OUTPUT_DIR}")"
mkdir -p "${CACHE_ROOT}" "${SPLIT_OUTPUT_DIR}"

cmd=(
    ${PYTHON_BIN} research/scripts/eval_ar_question_splits.py
    --run-dir "${RUNTIME_RUN_DIR}"
    --metrics-path "${RUNTIME_METRICS_PATH}"
    --output-dir "${RUNTIME_SPLIT_OUTPUT_DIR}"
    --n-splits "${NUM_SPLITS:-5}"
    --split-offset "${SPLIT_INDEX}"
    --single-split
    --seed "${SEED}"
    --min-seq-len "${MIN_SEQ_LEN}"
    --ar-device "${AR_DEVICE}"
    --ar-max-samples "${AR_MAX_SAMPLES}"
    --ar-train-steps "${AR_TRAIN_STEPS}"
    --ar-batch-size "${AR_BATCH_SIZE}"
    --ar-lr "${AR_LR}"
    --ar-d-model "${AR_D_MODEL}"
    --ar-n-layers "${AR_N_LAYERS}"
    --ar-n-heads "${AR_N_HEADS}"
    --ar-dropout "${AR_DROPOUT}"
    --ar-max-seq-len "${AR_MAX_SEQ_LEN}"
    --ar-min-buffer-samples "${AR_MIN_BUFFER_SAMPLES}"
)
if [[ "${USE_REASONING_SPAN}" == "1" ]]; then
    cmd+=(--use-reasoning-span)
fi
if [[ "${SPILL_TRAIN_SEQUENCES}" == "1" ]]; then
    cmd+=(--spill-train-sequences)
fi
if [[ "${KEEP_SPILLED_SEQUENCES}" == "1" ]]; then
    cmd+=(--keep-spilled-sequences)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
fi

log "Running AR split ${SPLIT_INDEX} on ${RUN_DIR}"
run_shell "$(build_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd "${cmd[@]}")"
