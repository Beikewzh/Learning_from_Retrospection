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
MIXED_QUESTIONS_ONLY="${MIXED_QUESTIONS_ONLY:-0}"
AR_DEVICE="${AR_DEVICE:-cuda}"
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-16000}"
AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-3000}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-16}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN:-4096}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-64}"
SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES:-1}"
KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES:-0}"
MAX_COMPONENTS="${MAX_COMPONENTS:-64}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
OVERWRITE="${OVERWRITE:-0}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${RUN_DIR}/analysis_parallel}"
AR_OUTPUT_DIR="${AR_OUTPUT_DIR:-${ANALYSIS_ROOT}/ar_model}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-${ANALYSIS_ROOT}/shards}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-${ANALYSIS_ROOT}/merged}"
AR_CHECKPOINT="${AR_OUTPUT_DIR}/checkpoints/latest.pt"

mkdir -p "${ANALYSIS_ROOT}" "${SHARD_OUTPUT_DIR}"

train_job_id=$(sbatch --parsable \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_DIR="${RUN_DIR}",CACHE_ROOT="${CACHE_ROOT}",AR_OUTPUT_DIR="${AR_OUTPUT_DIR}",USE_REASONING_SPAN="${USE_REASONING_SPAN}",MIXED_QUESTIONS_ONLY="${MIXED_QUESTIONS_ONLY}",AR_DEVICE="${AR_DEVICE}",AR_MAX_SAMPLES="${AR_MAX_SAMPLES}",AR_TRAIN_STEPS="${AR_TRAIN_STEPS}",AR_BATCH_SIZE="${AR_BATCH_SIZE}",AR_D_MODEL="${AR_D_MODEL}",AR_N_LAYERS="${AR_N_LAYERS}",AR_N_HEADS="${AR_N_HEADS}",AR_DROPOUT="${AR_DROPOUT}",AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN}",AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES}",MIN_SEQ_LEN="${MIN_SEQ_LEN}",SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES}",KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES}" \
  scripts/slurm/offline_train_ar_analysis.sbatch)

echo "Submitted AR train job: ${train_job_id}"

metric_job_ids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  jid=$(sbatch --parsable \
    --dependency=afterok:${train_job_id} \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_DIR="${RUN_DIR}",CACHE_ROOT="${CACHE_ROOT}",OUTPUT_DIR="${SHARD_OUTPUT_DIR}",AR_CHECKPOINT="${AR_CHECKPOINT}",AR_DEVICE="${AR_DEVICE}",USE_REASONING_SPAN="${USE_REASONING_SPAN}",MAX_COMPONENTS="${MAX_COMPONENTS}",MIN_SEQ_LEN="${MIN_SEQ_LEN}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${shard}",OVERWRITE="${OVERWRITE}" \
    scripts/slurm/offline_compute_ar_spectrum_metrics_shard.sbatch)
  metric_job_ids+=("${jid}")
  echo "Submitted metrics shard ${shard}: ${jid}"
done

deps=$(IFS=:; echo "${metric_job_ids[*]}")
merge_job_id=$(sbatch --parsable \
  --dependency=afterok:${deps} \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_DIR="${SHARD_OUTPUT_DIR}",OUTPUT_DIR="${MERGED_OUTPUT_DIR}" \
  scripts/slurm/offline_merge_ar_spectrum_metrics.sbatch)

echo "Submitted merge job: ${merge_job_id}"
echo "Merged metrics will be at: ${MERGED_OUTPUT_DIR}/metrics.jsonl"
