#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(3 4)
fi

RORQUAL_ACCOUNT="${RORQUAL_ACCOUNT:-rrg-bengioy-ad_gpu}"

submit_job() {
    local script_path="$1"
    shift
    local export_args="ALL"
    local arg
    for arg in "$@"; do
        export_args+=",${arg}"
    done
    echo "Submitting ${script_path} with ${export_args}"
    sbatch --account="${RORQUAL_ACCOUNT}" --export="${export_args}" "${script_path}"
}

for seed in "${SEEDS[@]}"; do
    submit_job scripts/rorqual/tutorial_math_grpo_lora_4gpu.sbatch \
        "SEED=${seed}" \
        "RUN_LABEL=final_vanilla_math_grpo_lora_4gpu_rorqual_seed${seed}"
    submit_job scripts/rorqual/tutorial_math_grpo_lora_4gpu_length_penalty.sbatch \
        "SEED=${seed}" \
        "LENGTH_PENALTY=1e-5" \
        "LENGTH_PENALTY_TAG=1e-5" \
        "RUN_LABEL=final_lenpen_math_grpo_lora_4gpu_rorqual_p1e-5_seed${seed}"
    submit_job scripts/rorqual/tutorial_math_prm_cpu_4gpu.sbatch \
        "SEED=${seed}" \
        "PRM_WEIGHT=0.1" \
        "PRM_WEIGHT_TAG=0p1" \
        "RUN_LABEL=final_prm_math_cpu_4gpu_rorqual_w0p1_seed${seed}"
    submit_job scripts/rorqual/tutorial_math_lears_lora_4gpu.sbatch \
        "SEED=${seed}" \
        "LEARS_ETA=0.005" \
        "LEARS_ETA_TAG=0p005" \
        "RUN_LABEL=final_lears_math_lora_4gpu_rorqual_eta0p005_seed${seed}"
done
