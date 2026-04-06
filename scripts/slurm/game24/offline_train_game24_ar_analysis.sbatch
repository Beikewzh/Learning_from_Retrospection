#!/bin/bash
#SBATCH --job-name=offline-game24-train-ar
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
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

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "RUN_DIR must be set" >&2
    exit 1
fi
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
OFFLINE_ENABLE_NV=1

AR_OUTPUT_DIR="${AR_OUTPUT_DIR:-${RUN_DIR}/analysis_parallel/ar_model}"
USE_REASONING_SPAN="${USE_REASONING_SPAN:-1}"
MIXED_QUESTIONS_ONLY="${MIXED_QUESTIONS_ONLY:-0}"
AR_DEVICE="${AR_DEVICE:-cuda}"
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-16000}"
AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-3000}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-16}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN:-4096}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-64}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"
SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES:-1}"
KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES:-0}"

offline_setup_container

RUNTIME_CACHE_ROOT="$(to_runtime_path "${CACHE_ROOT}")"
RUNTIME_RUN_DIR="$(to_runtime_path "${RUN_DIR}")"
RUNTIME_AR_OUTPUT_DIR="$(to_runtime_path "${AR_OUTPUT_DIR}")"
mkdir -p "${CACHE_ROOT}" "${AR_OUTPUT_DIR}"

cmd=(
    ${PYTHON_BIN} -m research.offline.game24.train_game24_ar_on_run
    --run-dir "${RUNTIME_RUN_DIR}"
    --output-dir "${RUNTIME_AR_OUTPUT_DIR}"
    --max-samples "${AR_MAX_SAMPLES}"
    --min-seq-len "${MIN_SEQ_LEN}"
    --device "${AR_DEVICE}"
    --train-steps "${AR_TRAIN_STEPS}"
    --batch-size "${AR_BATCH_SIZE}"
    --d-model "${AR_D_MODEL}"
    --n-layers "${AR_N_LAYERS}"
    --n-heads "${AR_N_HEADS}"
    --dropout "${AR_DROPOUT}"
    --max-seq-len "${AR_MAX_SEQ_LEN}"
    --min-buffer-samples "${AR_MIN_BUFFER_SAMPLES}"
)
if [[ "${USE_REASONING_SPAN}" == "1" ]]; then
    cmd+=(--use-reasoning-span)
fi
if [[ "${MIXED_QUESTIONS_ONLY}" == "1" ]]; then
    cmd+=(--mixed-questions-only)
fi
if [[ "${SPILL_TRAIN_SEQUENCES}" == "1" ]]; then
    cmd+=(--spill-train-sequences)
fi
if [[ "${KEEP_SPILLED_SEQUENCES}" == "1" ]]; then
    cmd+=(--keep-spilled-sequences)
fi

log "Training AR on ${RUN_DIR}"
run_shell "$(build_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd "${cmd[@]}")"
