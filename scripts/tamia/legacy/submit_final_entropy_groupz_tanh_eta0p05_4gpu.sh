#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4)
fi

submit_job() {
    local seed="$1"
    local export_args="ALL,SEED=${seed},ENTROPY_ETA=0.05,ENTROPY_ETA_TAG=0p05,RUN_LABEL=final_entropy_groupz_tanh_math_lora_4gpu_eta0p05_seed${seed}"
    echo "Submitting entropy group-zscore tanh eta=0.05 seed ${seed}"
    sbatch --export="${export_args}" scripts/tamia/tutorial_math_entropy_lora_4gpu_group_zscore_tanh.sbatch
}

for seed in "${SEEDS[@]}"; do
    submit_job "${seed}"
done
