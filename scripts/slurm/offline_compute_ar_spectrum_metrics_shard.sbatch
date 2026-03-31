#!/bin/bash
#SBATCH --job-name=offline-metrics
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=80G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/offline_common.sh"
offline_init_basic

RUN_DIR="${RUN_DIR:-}"
AR_CHECKPOINT="${AR_CHECKPOINT:-}"
if [[ -z "${RUN_DIR}" || -z "${AR_CHECKPOINT}" ]]; then
    echo "RUN_DIR and AR_CHECKPOINT must be set" >&2
    exit 1
fi
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
OFFLINE_ENABLE_NV=1

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/analysis_parallel/shards}"
USE_REASONING_SPAN="${USE_REASONING_SPAN:-1}"
AR_DEVICE="${AR_DEVICE:-cuda}"
MAX_COMPONENTS="${MAX_COMPONENTS:-64}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
OVERWRITE="${OVERWRITE:-0}"

offline_setup_container

RUNTIME_CACHE_ROOT="$(to_runtime_path "${CACHE_ROOT}")"
RUNTIME_RUN_DIR="$(to_runtime_path "${RUN_DIR}")"
RUNTIME_OUTPUT_DIR="$(to_runtime_path "${OUTPUT_DIR}")"
RUNTIME_AR_CHECKPOINT="$(to_runtime_path "${AR_CHECKPOINT}")"
mkdir -p "${CACHE_ROOT}" "${OUTPUT_DIR}"

cmd=(
    ${PYTHON_BIN} research/scripts/compute_ar_spectrum_metrics.py
    --run-dir "${RUNTIME_RUN_DIR}"
    --output-dir "${RUNTIME_OUTPUT_DIR}"
    --ar-checkpoint "${RUNTIME_AR_CHECKPOINT}"
    --ar-device "${AR_DEVICE}"
    --max-components "${MAX_COMPONENTS}"
    --min-seq-len "${MIN_SEQ_LEN}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${SHARD_INDEX}"
)
if [[ "${USE_REASONING_SPAN}" == "1" ]]; then
    cmd+=(--use-reasoning-span)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
fi

log "Computing metrics shard ${SHARD_INDEX}/${NUM_SHARDS} for ${RUN_DIR}"
run_shell "$(build_env_prefix "${RUNTIME_CACHE_ROOT}") && $(join_cmd "${cmd[@]}")"
