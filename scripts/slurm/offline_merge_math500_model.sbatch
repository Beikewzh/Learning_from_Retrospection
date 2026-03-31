#!/bin/bash
#SBATCH --job-name=offline-merge-one
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/offline_common.sh"
offline_init_basic

INPUT_ROOT="${INPUT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
OFFLINE_ENABLE_NV=0

if [[ -z "${INPUT_ROOT}" ]]; then
    echo "INPUT_ROOT must be set" >&2
    exit 1
fi

offline_setup_container

RUNTIME_INPUT_ROOT="$(to_runtime_path "${INPUT_ROOT}")"
RUNTIME_OUTPUT_DIR=""
if [[ -n "${OUTPUT_DIR}" ]]; then
    RUNTIME_OUTPUT_DIR="$(to_runtime_path "${OUTPUT_DIR}")"
fi

log "Repo root: ${REPO_ROOT}"
log "Input root: ${INPUT_ROOT}"
if [[ -n "${OUTPUT_DIR}" ]]; then
    log "Output dir: ${OUTPUT_DIR}"
fi

merge_cmd=(
    ${PYTHON_BIN} -m research.scripts.merge_offline_shards
    --input-root "${RUNTIME_INPUT_ROOT}"
    --overwrite
)
if [[ -n "${RUNTIME_OUTPUT_DIR}" ]]; then
    merge_cmd+=(--output-dir "${RUNTIME_OUTPUT_DIR}")
fi

run_shell "$(join_cmd "${merge_cmd[@]}")"
log "Done."
