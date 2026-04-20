#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
cd "${REPO_ROOT}"

MODEL_ID="${MODEL_ID:-}"
if [[ -z "${MODEL_ID}" ]]; then
    echo "MODEL_ID must be set, e.g. MODEL_ID=Qwen/Qwen3-4B" >&2
    exit 1
fi

sanitize_model_name() {
    local value="$1"
    value="${value//\//_}"
    value="${value//-/_}"
    value="${value//./_}"
    printf '%s' "${value,,}"
}

MODEL_TAG="${MODEL_TAG:-$(sanitize_model_name "${MODEL_ID}")}"
SEED="${SEED:-1}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"

DATA_JSON_SOURCE="${DATA_JSON_SOURCE:-${REPO_ROOT}/data/game24_reasoning/all_24_game_results_shuffled.json}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/game24_reasoning/game24_reasoning_limit${LIMIT:-500}.jsonl}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-${REPO_ROOT}/research/offline/game24/game24_prompt.jinja}"

LIMIT="${LIMIT:-500}"
NUM_SHARDS="${NUM_SHARDS:-10}"
NUM_SAMPLES_PER_QUESTION="${NUM_SAMPLES_PER_QUESTION:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
LATENT_DTYPE="${LATENT_DTYPE:-fp16}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
LAYER_INDEX="${LAYER_INDEX:--1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"
FLUSH_EVERY_N_SAMPLES="${FLUSH_EVERY_N_SAMPLES:-1}"
USE_REASONING_SPAN="${USE_REASONING_SPAN:-1}"

MAX_COMPONENTS="${MAX_COMPONENTS:-64}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4}"

MIXED_QUESTIONS_ONLY="${MIXED_QUESTIONS_ONLY:-0}"
AR_DEVICE="${AR_DEVICE:-cpu}"
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-20000}"
AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-5000}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-8}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN:-4096}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-64}"
SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES:-1}"
KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES:-0}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/offline_game24_temp1_k32}"
RUN_ROOT="${OUTPUT_BASE}/${MODEL_TAG}_limit${LIMIT}"
MERGED_RUN_DIR="${RUN_ROOT}/merged"
ANALYSIS_ROOT="${MERGED_RUN_DIR}/analysis_parallel"


echo "Submitting isolated Game24 end-to-end offline pipeline"
echo "  model_id=${MODEL_ID}"
echo "  model_tag=${MODEL_TAG}"
echo "  data_json_source=${DATA_JSON_SOURCE}"
echo "  data_path=${DATA_PATH}"
echo "  run_root=${RUN_ROOT}"
echo "  merged_run_dir=${MERGED_RUN_DIR}"
echo "  num_shards=${NUM_SHARDS}"
echo "  samples_per_question=${NUM_SAMPLES_PER_QUESTION}"

prep_jid=$(sbatch --parsable \
    --job-name="prep-g24-${MODEL_TAG}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",CACHE_ROOT="${CACHE_ROOT}",MODEL_ID="${MODEL_ID}",MODEL_TAG="${MODEL_TAG}",SEED="${SEED}",LIMIT="${LIMIT}",NUM_SHARDS=1,SHARD_INDEX=0,NUM_SAMPLES_PER_QUESTION="${NUM_SAMPLES_PER_QUESTION}",TEMPERATURE="${TEMPERATURE}",TOP_P="${TOP_P}",TOP_K="${TOP_K}",MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",LATENT_DTYPE="${LATENT_DTYPE}",TORCH_DTYPE="${TORCH_DTYPE}",LAYER_INDEX="${LAYER_INDEX}",OUTPUT_BASE="${OUTPUT_BASE}",OVERWRITE=0,RESUME=1,FLUSH_EVERY_N_SAMPLES="${FLUSH_EVERY_N_SAMPLES}",PREDOWNLOAD_ONLY=1,SKIP_PREPARE=0,SKIP_PREDOWNLOAD=0,DATA_JSON_SOURCE="${DATA_JSON_SOURCE}",DATA_PATH="${DATA_PATH}",PROMPT_TEMPLATE="${PROMPT_TEMPLATE}" \
    scripts/slurm/game24/offline_collect_game24_single_model.sbatch)
echo "Submitted prepare/predownload job: ${prep_jid}"

collection_job_ids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    jid=$(sbatch --parsable \
        --dependency="afterok:${prep_jid}" \
        --job-name="off-g24-${MODEL_TAG}-s$(printf '%02d' "${shard}")" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",CACHE_ROOT="${CACHE_ROOT}",MODEL_ID="${MODEL_ID}",MODEL_TAG="${MODEL_TAG}",SEED="${SEED}",LIMIT="${LIMIT}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${shard}",NUM_SAMPLES_PER_QUESTION="${NUM_SAMPLES_PER_QUESTION}",TEMPERATURE="${TEMPERATURE}",TOP_P="${TOP_P}",TOP_K="${TOP_K}",MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",LATENT_DTYPE="${LATENT_DTYPE}",TORCH_DTYPE="${TORCH_DTYPE}",LAYER_INDEX="${LAYER_INDEX}",OUTPUT_BASE="${OUTPUT_BASE}",OVERWRITE="${OVERWRITE}",RESUME="${RESUME}",FLUSH_EVERY_N_SAMPLES="${FLUSH_EVERY_N_SAMPLES}",PREDOWNLOAD_ONLY=0,SKIP_PREPARE=1,SKIP_PREDOWNLOAD=1,DATA_JSON_SOURCE="${DATA_JSON_SOURCE}",DATA_PATH="${DATA_PATH}",PROMPT_TEMPLATE="${PROMPT_TEMPLATE}" \
        scripts/slurm/game24/offline_collect_game24_single_model.sbatch)
    collection_job_ids+=("${jid}")
    echo "Submitted collection shard ${shard}: ${jid}"
done

collection_deps=$(IFS=:; echo "${collection_job_ids[*]}")
collection_merge_jid=$(sbatch --parsable \
    --job-name="merge-g24-${MODEL_TAG}" \
    --dependency="afterok:${collection_deps}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_ROOT="${RUN_ROOT}" \
    scripts/slurm/game24/offline_merge_game24_model.sbatch)
echo "Submitted collection merge: ${collection_merge_jid}"

ar_output_dir="${ANALYSIS_ROOT}/ar_model"
ar_shard_dir="${ANALYSIS_ROOT}/shards"
ar_merged_dir="${ANALYSIS_ROOT}/merged"
ar_checkpoint="${ar_output_dir}/checkpoints/latest.pt"

ar_train_jid=$(sbatch --parsable \
    --job-name="artrain-g24-${MODEL_TAG}" \
    --dependency="afterok:${collection_merge_jid}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_DIR="${MERGED_RUN_DIR}",CACHE_ROOT="${CACHE_ROOT}",AR_OUTPUT_DIR="${ar_output_dir}",USE_REASONING_SPAN="${USE_REASONING_SPAN}",MIXED_QUESTIONS_ONLY="${MIXED_QUESTIONS_ONLY}",AR_DEVICE="${AR_DEVICE}",AR_MAX_SAMPLES="${AR_MAX_SAMPLES}",AR_TRAIN_STEPS="${AR_TRAIN_STEPS}",AR_BATCH_SIZE="${AR_BATCH_SIZE}",AR_D_MODEL="${AR_D_MODEL}",AR_N_LAYERS="${AR_N_LAYERS}",AR_N_HEADS="${AR_N_HEADS}",AR_DROPOUT="${AR_DROPOUT}",AR_MAX_SEQ_LEN="${AR_MAX_SEQ_LEN}",AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES}",MIN_SEQ_LEN="${MIN_SEQ_LEN}",SPILL_TRAIN_SEQUENCES="${SPILL_TRAIN_SEQUENCES}",KEEP_SPILLED_SEQUENCES="${KEEP_SPILLED_SEQUENCES}" \
    scripts/slurm/game24/offline_train_game24_ar_analysis.sbatch)
echo "Submitted AR train: ${ar_train_jid}"

metric_job_ids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    jid=$(sbatch --parsable \
        --dependency="afterok:${ar_train_jid}" \
        --job-name="armet-g24-${MODEL_TAG}-s$(printf '%02d' "${shard}")" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_DIR="${MERGED_RUN_DIR}",CACHE_ROOT="${CACHE_ROOT}",OUTPUT_DIR="${ar_shard_dir}",AR_CHECKPOINT="${ar_checkpoint}",AR_DEVICE="${AR_DEVICE}",USE_REASONING_SPAN="${USE_REASONING_SPAN}",MAX_COMPONENTS="${MAX_COMPONENTS}",MIN_SEQ_LEN="${MIN_SEQ_LEN}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${shard}",OVERWRITE="${OVERWRITE}" \
        scripts/slurm/game24/offline_compute_game24_ar_spectrum_metrics_shard.sbatch)
    metric_job_ids+=("${jid}")
    echo "Submitted AR metrics shard ${shard}: ${jid}"
done

metric_deps=$(IFS=:; echo "${metric_job_ids[*]}")
ar_merge_jid=$(sbatch --parsable \
    --job-name="merge-ar-g24-${MODEL_TAG}" \
    --dependency="afterok:${metric_deps}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_DIR="${ar_shard_dir}",OUTPUT_DIR="${ar_merged_dir}" \
    scripts/slurm/game24/offline_merge_game24_metrics.sbatch)
echo "Submitted AR metrics merge: ${ar_merge_jid}"

echo
echo "Done submitting Game24 pipeline."
echo "Collection merged run:"
echo "  ${MERGED_RUN_DIR}"
echo "Canonical merged metrics:"
echo "  ${ar_merged_dir}/metrics.jsonl"
