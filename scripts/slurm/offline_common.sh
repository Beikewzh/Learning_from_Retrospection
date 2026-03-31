#!/usr/bin/env bash

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

offline_init_basic() {
    REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
    cd "${REPO_ROOT}"
    mkdir -p slurm_logs
    RUNTIME="${RUNTIME:-/workspace}"
}

offline_setup_container() {
    USE_SINGULARITY="${USE_SINGULARITY:-1}"
    SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
    OFFLINE_ENABLE_NV="${OFFLINE_ENABLE_NV:-1}"
    CONTAINER_BIN="${CONTAINER_BIN:-}"

    if [[ "${USE_SINGULARITY}" != "1" ]]; then
        return
    fi

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
        echo "No singularity/apptainer found" >&2
        exit 1
    fi
    if [[ ! -f "${SIF_PATH}" ]]; then
        echo "Missing container image: ${SIF_PATH}" >&2
        exit 1
    fi
}

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
        local nv_args=()
        if [[ "${OFFLINE_ENABLE_NV}" == "1" ]]; then
            nv_args+=(--nv)
        fi
        "${CONTAINER_BIN}" exec "${nv_args[@]}" --cleanenv \
            --bind "${REPO_ROOT}:${RUNTIME}" \
            "${SIF_PATH}" \
            bash -lc "cd ${RUNTIME} && ${cmd}"
    else
        bash -lc "cd ${REPO_ROOT} && ${cmd}"
    fi
}
