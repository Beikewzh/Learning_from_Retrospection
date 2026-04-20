#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4)
fi

bash scripts/tamia/final/submit_vanilla_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_correct_length_group_penalty_alpha0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_l1_exact_alpha0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_l1_max_alpha0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_lears_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_lears_groupz_tanh_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_length_groupz_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_length_groupz_tanh_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_entropy_group_zscore_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/final/submit_entropy_groupz_tanh_eta0p01_4gpu.sh "${SEEDS[@]}"
