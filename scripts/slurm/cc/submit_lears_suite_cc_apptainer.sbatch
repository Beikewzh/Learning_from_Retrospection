#!/bin/bash
#SBATCH --job-name=lears-suite-cc
#SBATCH --partition=gpubase_bynode_b5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/slurm_logs"

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
if command -v module >/dev/null 2>&1; then
    module load apptainer >/dev/null 2>&1 || true
    module load apptainer/1 >/dev/null 2>&1 || true
fi

SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_ROOT}/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_ROOT}/.apptainer_tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export USE_SINGULARITY=1
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${APPTAINER_TMPDIR}}"

exec bash "${REPO_ROOT}/scripts/slurm/submit_lears_suite.sbatch"
