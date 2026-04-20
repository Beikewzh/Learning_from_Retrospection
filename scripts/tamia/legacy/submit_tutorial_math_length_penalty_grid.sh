#!/bin/bash

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"

SCRIPT="scripts/tamia/tutorial_math_grpo_lora_4gpu_length_penalty.sbatch"

PENALTIES=(
  "1e-5"
  "5e-5"
  "1e-4"
  "5e-4"
  "1e-3"
)

for penalty in "${PENALTIES[@]}"; do
  penalty_tag="${penalty//./p}"
  echo "Submitting length penalty=${penalty}"
  sbatch --export=ALL,LENGTH_PENALTY="${penalty}",LENGTH_PENALTY_TAG="${penalty_tag}",LENGTH_THRESHOLD="512" "${SCRIPT}"
done
