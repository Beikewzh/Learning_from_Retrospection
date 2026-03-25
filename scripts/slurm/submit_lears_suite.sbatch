#!/bin/bash
#SBATCH --job-name=lears-suite
#SBATCH --account=REPLACE_ME
#SBATCH --partition=REPLACE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --gres=gpu:REPLACE_ME:2
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#
# Thin Slurm entrypoint: edit #SBATCH above (or override at submit time), then put experiment
# knobs in scripts/slurm/suites/*.sh and point SUITE= there.
#
# Examples:
#   sbatch scripts/slurm/submit_lears_suite.sbatch
#   sbatch --export=ALL,SUITE=scripts/slurm/suites/math12k_2gpu_offline_short_steps.sh \
#     scripts/slurm/submit_lears_suite.sbatch
#
# Compute Canada: load Apptainer before sbatch if your site requires it, e.g.
#   module load StdEnv/2023 apptainer/1.3.4
#
# For an Alliance-style wrapper that loads modules inside the batch file, see:
#   scripts/slurm/cc/submit_lears_suite_cc_apptainer.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/slurm_logs"

SUITE="${SUITE:-scripts/slurm/suites/math12k_2gpu_offline.sh}"
if [[ "${SUITE}" != /* ]]; then
    SUITE="${REPO_ROOT}/${SUITE}"
fi
if [[ ! -f "${SUITE}" ]]; then
    echo "SUITE file not found: ${SUITE}" >&2
    echo "Set SUITE to a path under scripts/slurm/suites/ (or sbatch --export=ALL,SUITE=...)." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${SUITE}"
exec bash "${REPO_ROOT}/scripts/slurm/lears_full_train.sbatch"
