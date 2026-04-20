#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4)
fi

submit_for_all_seeds() {
    local launcher="$1"
    local label_prefix="$2"
    shift 2
    local extra_exports=("$@")

    for seed in "${SEEDS[@]}"; do
        local export_args="ALL,SEED=${seed},RUN_LABEL=${label_prefix}_seed${seed}"
        for export_kv in "${extra_exports[@]}"; do
            export_args+=",${export_kv}"
        done
        echo "Submitting ${label_prefix} seed ${seed}"
        sbatch --export="${export_args}" "${launcher}"
    done
}

submit_for_all_seeds scripts/tamia/final/game24_grpo_lora_4gpu.sbatch \
    game24_vanilla_grpo_lora_4gpu
submit_for_all_seeds scripts/tamia/final/game24_grpo_lora_4gpu_correct_length_group_penalty.sbatch \
    game24_corrlen_grpo_lora_4gpu_a0p01 \
    LENGTH_ALPHA=0.01 LENGTH_ALPHA_TAG=0p01
submit_for_all_seeds scripts/tamia/final/game24_grpo_lora_4gpu_l1_exact.sbatch \
    game24_l1exact_grpo_lora_4gpu_a0p01_t512 \
    L1_ALPHA=0.01 L1_ALPHA_TAG=0p01 TARGET_LENGTH=512
submit_for_all_seeds scripts/tamia/final/game24_grpo_lora_4gpu_l1_max.sbatch \
    game24_l1max_grpo_lora_4gpu_a0p01_t512_d0p5 \
    L1_ALPHA=0.01 L1_ALPHA_TAG=0p01 TARGET_LENGTH=512 DELTA=0.5 DELTA_TAG=0p5
submit_for_all_seeds scripts/tamia/final/game24_lears_lora_4gpu_group_zscore.sbatch \
    game24_lears_groupz_lora_4gpu_eta0p01 \
    LEARS_ETA=0.01 LEARS_ETA_TAG=0p01
submit_for_all_seeds scripts/tamia/final/game24_lears_lora_4gpu_group_zscore_tanh.sbatch \
    game24_lears_groupz_tanh_lora_4gpu_eta0p01 \
    LEARS_ETA=0.01 LEARS_ETA_TAG=0p01
submit_for_all_seeds scripts/tamia/final/game24_length_lora_4gpu_group_zscore.sbatch \
    game24_length_groupz_lora_4gpu_eta0p01 \
    LENGTH_ETA=0.01 LENGTH_ETA_TAG=0p01
submit_for_all_seeds scripts/tamia/final/game24_length_lora_4gpu_group_zscore_tanh.sbatch \
    game24_length_groupz_tanh_lora_4gpu_eta0p01 \
    LENGTH_ETA=0.01 LENGTH_ETA_TAG=0p01
