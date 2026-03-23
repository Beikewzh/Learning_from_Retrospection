#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
    echo "RUN_DIR must be set" >&2
    exit 1
fi
METRICS_PATH="${METRICS_PATH:-${RUN_DIR}/spectrum_parallel/merged/metrics.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-${RUN_DIR}/ar_split_eval_parallel}"
NUM_SPLITS="${NUM_SPLITS:-5}"
SEED="${SEED:-1}"
USE_REASONING_SPAN="${USE_REASONING_SPAN:-1}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"
AR_DEVICE="${AR_DEVICE:-cpu}"
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-1000}"
AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-100}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-8}"
AR_LR="${AR_LR:-1e-4}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN:-4096}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-64}"
SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES:-1}"
KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES:-0}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
OVERWRITE="${OVERWRITE:-0}"
mkdir -p "${OUTPUT_BASE}"

job_ids=()
for ((split=0; split<NUM_SPLITS; split++)); do
  jid=$(sbatch --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_DIR="${RUN_DIR}",METRICS_PATH="${METRICS_PATH}",OUTPUT_BASE="${OUTPUT_BASE}",NUM_SPLITS="${NUM_SPLITS}",SPLIT_INDEX="${split}",SEED="${SEED}",USE_REASONING_SPAN="${USE_REASONING_SPAN}",MIN_SEQ_LEN="${MIN_SEQ_LEN}",AR_DEVICE="${AR_DEVICE}",AR_MAX_SAMPLES="${AR_MAX_SAMPLES}",AR_TRAIN_STEPS="${AR_TRAIN_STEPS}",AR_BATCH_SIZE="${AR_BATCH_SIZE}",AR_LR="${AR_LR}",AR_D_MODEL="${AR_D_MODEL}",AR_N_LAYERS="${AR_N_LAYERS}",AR_N_HEADS="${AR_N_HEADS}",AR_DROPOUT="${AR_DROPOUT}",AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN}",AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES}",SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES}",KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES}",CACHE_ROOT="${CACHE_ROOT}",OVERWRITE="${OVERWRITE}" \
    scripts/slurm/offline_eval_ar_question_split.sbatch)
  job_ids+=("${jid}")
  echo "Submitted split ${split}: ${jid}"
done

deps=$(IFS=:; echo "${job_ids[*]}")
merge_job_id=$(sbatch --parsable \
  --dependency=afterok:${deps} \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_DIR="${OUTPUT_BASE}",OUTPUT_DIR="${OUTPUT_BASE}/merged" \
  scripts/slurm/offline_merge_ar_split_eval.sbatch)

echo "Submitted merge job: ${merge_job_id}"
echo "Merged split-eval results will be at: ${OUTPUT_BASE}/merged/split_results.jsonl"
