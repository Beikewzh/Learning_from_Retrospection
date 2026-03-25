#!/bin/bash
#SBATCH --job-name=rorqual-baseline-ar
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#
# Rorqual: 1 node × 4 GPUs. For 8 GPUs: sbatch --gres=gpu:8 --cpus-per-task=32 --mem=120G ...
# Qwen3-4B + AR training + logging/checkpoints; intrinsic eta=0 (math-only RL advantages).
# From repo root: sbatch scripts/rorqual/submit_baseline_ar_logging_only.sbatch

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export SUITE="${REPO_ROOT}/scripts/rorqual/suites/baseline_qwen4b_ar_logging_only.sh"
exec bash "${REPO_ROOT}/scripts/slurm/submit_lears_suite.sbatch"
