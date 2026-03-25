#!/bin/bash
#SBATCH --job-name=rorqual-lears-intrinsic
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#
# Rorqual: 1 node × 4 GPUs (matches tensor_parallel_size=2 → dp=2). lears_full_train auto-sets
# trainer.n_gpus_per_node from the Slurm GPU count.
# For 8 GPUs / other sites: sbatch --gres=gpu:8 --cpus-per-task=32 --mem=120G ...
# Qwen3-4B + AR + full intrinsic (eta + lambdas like lears_default).
# From repo root: sbatch scripts/rorqual/submit_lears_intrinsic.sbatch

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export SUITE="${REPO_ROOT}/scripts/rorqual/suites/qwen4b_lears_intrinsic.sh"
exec bash "${REPO_ROOT}/scripts/slurm/submit_lears_suite.sbatch"
