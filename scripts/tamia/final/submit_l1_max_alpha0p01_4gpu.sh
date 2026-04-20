#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4)
fi

submit_job() {
    local seed="$1"
    local export_args="ALL,SEED=${seed},L1_ALPHA=0.01,L1_ALPHA_TAG=0p01,TARGET_LENGTH=512,DELTA=0.5,DELTA_TAG=0p5,RUN_LABEL=final_l1max_math_grpo_lora_4gpu_a0p01_t512_d0p5_seed${seed}"
    echo "Submitting L1-Max alpha=0.01 seed ${seed}"
    sbatch --export="${export_args}" scripts/tamia/final/math_grpo_lora_4gpu_l1_max.sbatch
}

for seed in "${SEEDS[@]}"; do
    submit_job "${seed}"
done
