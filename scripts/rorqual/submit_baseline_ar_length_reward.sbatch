#!/bin/bash
#SBATCH --job-name=rorqual-ar-length
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#
# Rorqual: 1 node × 4 H100 (NVLink). tensor_parallel_size=2 → dp=2.
# Qwen3-4B + length-penalised GRPO reward (reward / T) + AR training/logging; intrinsic eta=0.
# From repo root: sbatch scripts/rorqual/submit_baseline_ar_length_reward.sbatch

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export SUITE="${REPO_ROOT}/scripts/rorqual/suites/baseline_qwen4b_ar_length_reward.sh"
exec bash "${REPO_ROOT}/scripts/slurm/submit_lears_suite.sbatch"
