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
    local export_args="ALL,SEED=${seed},LENGTH_ALPHA=0.01,LENGTH_ALPHA_TAG=0p01,RUN_LABEL=final_corrlen_math_grpo_lora_4gpu_a0p01_seed${seed}"
    echo "Submitting correct-only group-length baseline alpha=0.01 seed ${seed}"
    sbatch --export="${export_args}" scripts/tamia/final/math_grpo_lora_4gpu_correct_length_group_penalty.sbatch
}

for seed in "${SEEDS[@]}"; do
    submit_job "${seed}"
done
