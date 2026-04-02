#!/bin/bash

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"

SCRIPT="scripts/tamia/tutorial_math_lears_lora_4gpu.sbatch"

# Small 5-point grid around the current default eta=0.02.
ETAS=(
  "0.01"
  "0.02"
  "0.05"
  "0.1"
  "0.2"
)

for eta in "${ETAS[@]}"; do
  eta_tag="${eta//./p}"
  echo "Submitting LeaRS eta=${eta}"
  sbatch --export=ALL,LEARS_ETA="${eta}",LEARS_ETA_TAG="${eta_tag}" "${SCRIPT}"
done
