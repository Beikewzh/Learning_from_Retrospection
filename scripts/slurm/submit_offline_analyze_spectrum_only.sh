#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "RUN_DIR must be set, e.g. RUN_DIR=${REPO_ROOT}/outputs/offline_math500_temp1_k32/qwen_qwen3_4b_limit500/merged" >&2
    exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-10}"
USE_REASONING_SPAN="${USE_REASONING_SPAN:-1}"
MAX_COMPONENTS="${MAX_COMPONENTS:-64}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
OVERWRITE="${OVERWRITE:-0}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${RUN_DIR}/spectrum_parallel}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-${ANALYSIS_ROOT}/shards}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-${ANALYSIS_ROOT}/merged}"
mkdir -p "${ANALYSIS_ROOT}" "${SHARD_OUTPUT_DIR}"

job_ids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  jid=$(sbatch --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_DIR="${RUN_DIR}",CACHE_ROOT="${CACHE_ROOT}",OUTPUT_DIR="${SHARD_OUTPUT_DIR}",USE_REASONING_SPAN="${USE_REASONING_SPAN}",MAX_COMPONENTS="${MAX_COMPONENTS}",MIN_SEQ_LEN="${MIN_SEQ_LEN}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${shard}",OVERWRITE="${OVERWRITE}" \
    scripts/slurm/offline_compute_spectrum_metrics_shard.sbatch)
  job_ids+=("${jid}")
  echo "Submitted spectrum shard ${shard}: ${jid}"
done

deps=$(IFS=:; echo "${job_ids[*]}")
merge_job_id=$(sbatch --parsable \
  --dependency=afterok:${deps} \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_DIR="${SHARD_OUTPUT_DIR}",OUTPUT_DIR="${MERGED_OUTPUT_DIR}" \
  scripts/slurm/offline_merge_ar_spectrum_metrics.sbatch)

echo "Submitted merge job: ${merge_job_id}"
echo "Merged spectrum metrics will be at: ${MERGED_OUTPUT_DIR}/metrics.jsonl"
