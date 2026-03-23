#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
TULU_MODEL="${TULU_MODEL:-allenai/Llama-3.1-Tulu-3-8B}"
SEED="${SEED:-1}"
NUM_SHARDS="${NUM_SHARDS:-10}"
LIMIT="${LIMIT:-500}"

sanitize_model_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value//-/_}"
    value="${value//./_}"
    printf '%s' "${value,,}"
}

QWEN_TAG="${QWEN_TAG:-$(sanitize_model_name "${QWEN_MODEL}")}"
TULU_TAG="${TULU_TAG:-$(sanitize_model_name "${TULU_MODEL}")}"

submit_model_shards() {
    local model_id="$1"
    local model_tag="$2"
    local seed_base="$3"
    local shard
    local job_ids=()
    local job_id
    for ((shard=0; shard<NUM_SHARDS; shard++)); do
        echo "Submitting ${model_id} shard ${shard}/${NUM_SHARDS}"
        job_id="$(sbatch --parsable --job-name="off-${model_tag}-s$(printf '%02d' "${shard}")" \
            --export=ALL,MODEL_ID="${model_id}",MODEL_TAG="${model_tag}",SEED="${seed_base}",LIMIT="${LIMIT}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${shard}" \
            scripts/slurm/offline_collect_math500_single_model.sbatch)"
        echo "  job_id=${job_id}"
        job_ids+=("${job_id}")
    done

    local dependency
    dependency="$(IFS=:; echo "${job_ids[*]}")"
    local input_root="${REPO_ROOT}/outputs/offline_math500_temp1_k32/${model_tag}_limit${LIMIT}"
    echo "Submitting merge job for ${model_id} afterok:${dependency}"
    sbatch --job-name="merge-${model_tag}" \
        --dependency="afterok:${dependency}" \
        --export=ALL,INPUT_ROOT="${input_root}" \
        scripts/slurm/offline_merge_math500_model.sbatch
}

submit_model_shards "${QWEN_MODEL}" "${QWEN_TAG}" "${SEED}"
submit_model_shards "${TULU_MODEL}" "${TULU_TAG}" "$((SEED + 100000))"
