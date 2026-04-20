#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2)
fi

FIR_ACCOUNT="${FIR_ACCOUNT:-rrg-bengioy-ad_gpu}"

submit_job() {
    local seed="$1"
    local export_args="ALL,SEED=${seed},LEARS_ETA=0.005,LEARS_ETA_TAG=0p005,RUN_LABEL=final_lears_math_lora_4gpu_fir_eta0p005_seed${seed}"
    echo "Submitting Fir LeaRS eta=0.005 seed ${seed}"
    sbatch --account="${FIR_ACCOUNT}" --export="${export_args}" scripts/fir/tutorial_math_lears_lora_4gpu.sbatch
}

for seed in "${SEEDS[@]}"; do
    submit_job "${seed}"
done
