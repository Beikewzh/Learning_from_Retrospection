#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"

ACTOR_MODEL_ID="${ACTOR_MODEL_ID:-Qwen/Qwen3-4B}"
ACTOR_MODEL_DIR="${ACTOR_MODEL_DIR:-models/Qwen__Qwen3-4B}"
PRM_MODEL_ID="${PRM_MODEL_ID:-Qwen/Qwen2.5-Math-PRM-7B}"
PRM_MODEL_DIR="${PRM_MODEL_DIR:-models/Qwen__Qwen2.5-Math-PRM-7B}"

cd "${REPO_ROOT}"
mkdir -p models data slurm_logs

module load StdEnv/2023 apptainer/1.3.5

apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace" \
  "${SIF_PATH}" \
  bash -lc "
    set -euo pipefail
    cd /workspace

    mkdir -p \"$(dirname "${ACTOR_MODEL_DIR}")\" \"$(dirname "${PRM_MODEL_DIR}")\" data/math12k data/math500

    huggingface-cli download ${ACTOR_MODEL_ID} \
      --local-dir ${ACTOR_MODEL_DIR} \
      --exclude '*.gguf'

    huggingface-cli download ${PRM_MODEL_ID} \
      --local-dir ${PRM_MODEL_DIR} \
      --exclude '*.gguf'

    python3 - <<'PY'
from datasets import load_dataset
import os

os.makedirs('data/math12k', exist_ok=True)
os.makedirs('data/math500', exist_ok=True)

ds = load_dataset('hiyouga/math12k', split='train')
ds.to_parquet('data/math12k/train.parquet')
print(f'math12k: {len(ds)} rows saved')

ds2 = load_dataset('HuggingFaceH4/MATH-500', split='test')
ds2.to_parquet('data/math500/test.parquet')
print(f'MATH-500: {len(ds2)} rows saved')
PY
  "

echo "Downloaded offline assets into:"
echo "  ${REPO_ROOT}/${ACTOR_MODEL_DIR}"
echo "  ${REPO_ROOT}/${PRM_MODEL_DIR}"
echo "  ${REPO_ROOT}/data/math12k/train.parquet"
echo "  ${REPO_ROOT}/data/math500/test.parquet"
