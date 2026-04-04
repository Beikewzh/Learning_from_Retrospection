#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4)
fi

FIR_ACCOUNT="${FIR_ACCOUNT:?Set FIR_ACCOUNT to your Fir Slurm account before submitting.}"

submit_job() {
    local script_path="$1"
    shift
    local export_args="ALL"
    local arg
    for arg in "$@"; do
        export_args+=",${arg}"
    done
    echo "Submitting ${script_path} with ${export_args}"
    sbatch --account="${FIR_ACCOUNT}" --export="${export_args}" "${script_path}"
}

for seed in "${SEEDS[@]}"; do
    submit_job scripts/fir/tutorial_math_grpo_lora_4gpu.sbatch \
        "SEED=${seed}" \
        "RUN_LABEL=final_vanilla_math_grpo_lora_4gpu_fir_seed${seed}"
    submit_job scripts/fir/tutorial_math_grpo_lora_4gpu_length_penalty.sbatch \
        "SEED=${seed}" \
        "LENGTH_PENALTY=1e-5" \
        "LENGTH_PENALTY_TAG=1e-5" \
        "RUN_LABEL=final_lenpen_math_grpo_lora_4gpu_fir_p1e-5_seed${seed}"
    submit_job scripts/fir/tutorial_math_prm_4gpu.sbatch \
        "SEED=${seed}" \
        "PRM_WEIGHT=0.1" \
        "PRM_WEIGHT_TAG=0p1" \
        "RUN_LABEL=final_prm_math_4gpu_fir_w0p1_seed${seed}"
    submit_job scripts/fir/tutorial_math_lears_lora_4gpu.sbatch \
        "SEED=${seed}" \
        "LEARS_ETA=0.05" \
        "LEARS_ETA_TAG=0p05" \
        "RUN_LABEL=final_lears_math_lora_4gpu_fir_eta0p05_seed${seed}"
done
