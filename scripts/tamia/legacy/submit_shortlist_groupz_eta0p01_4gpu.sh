#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(1 2 3 4)
fi

bash scripts/tamia/submit_final_lears_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/submit_final_lears_groupz_tanh_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/submit_final_entropy_group_zscore_eta0p01_4gpu.sh "${SEEDS[@]}"
bash scripts/tamia/submit_final_entropy_groupz_tanh_eta0p01_4gpu.sh "${SEEDS[@]}"
