#!/bin/bash
#SBATCH --job-name=offline-merge-ar-split
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/offline_common.sh"
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
run_shell "$(join_cmd ${PYTHON_BIN} research/scripts/merge_split_eval_results.py --input-dir "${RUNTIME_INPUT_DIR}" --output-dir "${RUNTIME_OUTPUT_DIR}" --overwrite)"
