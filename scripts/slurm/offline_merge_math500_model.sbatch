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

INPUT_ROOT="${INPUT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
CONTAINER_BIN=""

if [[ -z "${INPUT_ROOT}" ]]; then
    echo "INPUT_ROOT must be set" >&2
    exit 1
fi

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
        echo "No singularity/apptainer found." >&2
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
