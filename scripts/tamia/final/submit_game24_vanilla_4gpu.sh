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
    local export_args="ALL,SEED=${seed},RUN_LABEL=game24_vanilla_grpo_lora_4gpu_seed${seed}"
    echo "Submitting Game of 24 vanilla seed ${seed}"
    sbatch --export="${export_args}" scripts/tamia/final/game24_grpo_lora_4gpu.sbatch
}

for seed in "${SEEDS[@]}"; do
    submit_job "${seed}"
done
