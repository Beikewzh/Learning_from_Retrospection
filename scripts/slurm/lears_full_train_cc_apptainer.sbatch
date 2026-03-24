#!/bin/bash
#SBATCH --job-name=lears-online-full-cc
#SBATCH --partition=gpubase_bynode_b5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

# Resolve repo root from submit directory by default.
if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/slurm_logs"

# Ensure module command exists in non-login batch shells.
if [[ -f /etc/profile ]]; then
    set +u
    # shellcheck source=/dev/null
    source /etc/profile >/dev/null 2>&1 || true
    set -u
fi
if ! command -v module >/dev/null 2>&1; then
    set +u
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
    # shellcheck source=/dev/null
    source /usr/share/modules/init/bash >/dev/null 2>&1 || true
    set -u
fi

# Compute Canada: use apptainer explicitly.
if command -v module >/dev/null 2>&1; then
    module load apptainer >/dev/null 2>&1 || true
    module load apptainer/1 >/dev/null 2>&1 || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer not found. Load an apptainer module or add it to PATH." >&2
    exit 1
fi

# Use apptainer cache on scratch.
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_ROOT}/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_ROOT}/.apptainer_tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# Keep backward compatibility with the existing launcher env names.
export USE_SINGULARITY=1
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${APPTAINER_TMPDIR}}"

exec bash "${REPO_ROOT}/scripts/slurm/lears_full_train.sbatch"
