#!/bin/bash
#SBATCH --job-name=offline-merge-metrics
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

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

INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "${INPUT_DIR}" ]]; then
    echo "INPUT_DIR must be set" >&2
    exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
CONTAINER_BIN=""

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${INPUT_DIR}/merged"
fi

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

RUNTIME_INPUT_DIR="$(to_runtime_path "${INPUT_DIR}")"
RUNTIME_OUTPUT_DIR="$(to_runtime_path "${OUTPUT_DIR}")"
run_shell "$(join_cmd ${PYTHON_BIN} research/scripts/merge_analysis_metrics.py --input-dir "${RUNTIME_INPUT_DIR}" --output-dir "${RUNTIME_OUTPUT_DIR}" --overwrite)"
