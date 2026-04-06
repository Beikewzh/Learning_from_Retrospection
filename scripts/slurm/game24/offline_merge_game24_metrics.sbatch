#!/bin/bash
#SBATCH --job-name=offline-game24-merge-metrics
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
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

INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "${INPUT_DIR}" ]]; then
    echo "INPUT_DIR must be set" >&2
    exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
OFFLINE_ENABLE_NV=0

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${INPUT_DIR}/merged"
fi

offline_setup_container

RUNTIME_INPUT_DIR="$(to_runtime_path "${INPUT_DIR}")"
RUNTIME_OUTPUT_DIR="$(to_runtime_path "${OUTPUT_DIR}")"
run_shell "$(join_cmd ${PYTHON_BIN} -m research.offline.game24.merge_game24_analysis_metrics --input-dir "${RUNTIME_INPUT_DIR}" --output-dir "${RUNTIME_OUTPUT_DIR}" --overwrite)"
