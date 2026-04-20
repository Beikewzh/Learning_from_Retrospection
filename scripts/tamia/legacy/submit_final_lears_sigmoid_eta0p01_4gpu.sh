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
    local export_args="ALL,SEED=${seed},LEARS_ETA=0.01,LEARS_ETA_TAG=0p01,RUN_LABEL=final_lears_sigmoid_math_lora_4gpu_eta0p01_seed${seed}"
    echo "Submitting LeaRS sigmoid eta=0.01 seed ${seed}"
    sbatch --export="${export_args}" scripts/tamia/tutorial_math_lears_lora_4gpu_sigmoid.sbatch
}

for seed in "${SEEDS[@]}"; do
    submit_job "${seed}"
done
