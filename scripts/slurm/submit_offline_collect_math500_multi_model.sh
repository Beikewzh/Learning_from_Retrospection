#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
TULU_MODEL="${TULU_MODEL:-allenai/Llama-3.1-Tulu-3-8B}"
SEED="${SEED:-1}"

sanitize_model_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value//-/_}"
    value="${value//./_}"
    printf '%s' "${value,,}"
}

QWEN_TAG="${QWEN_TAG:-$(sanitize_model_name "${QWEN_MODEL}")}"
TULU_TAG="${TULU_TAG:-$(sanitize_model_name "${TULU_MODEL}")}"

echo "Submitting ${QWEN_MODEL}"
sbatch --job-name="offline-${QWEN_TAG}" \
    --export=ALL,MODEL_ID="${QWEN_MODEL}",MODEL_TAG="${QWEN_TAG}",SEED="${SEED}" \
    scripts/slurm/offline_collect_math500_single_model.sbatch

echo "Submitting ${TULU_MODEL}"
sbatch --job-name="offline-${TULU_TAG}" \
    --export=ALL,MODEL_ID="${TULU_MODEL}",MODEL_TAG="${TULU_TAG}",SEED="$((SEED + 100000))" \
    scripts/slurm/offline_collect_math500_single_model.sbatch
