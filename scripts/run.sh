#!/bin/bash
# Simple training launcher. Run from repo root:
#   bash scripts/run.sh                          # defaults
#   TRAINER_N_GPUS_PER_NODE=4 bash scripts/run.sh
#   MAX_STEPS=1868 bash scripts/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-examples/research/rorqual/lears_intrinsic_math12k.yaml}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen__Qwen3-4B}"
N_GPUS="${TRAINER_N_GPUS_PER_NODE:-2}"
# Leave MAX_STEPS unset to use trainer.total_epochs from the YAML (epoch mode).
# Set MAX_STEPS=N to override with a fixed step count.
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-256}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lears_run_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-checkpoints/lears/${EXPERIMENT_NAME}}"

# Load apptainer if not already available
if ! command -v apptainer >/dev/null 2>&1; then
    module load StdEnv/2023 apptainer/1.3.5
fi

exec apptainer exec --nv \
    --bind "${REPO_ROOT}:/workspace" \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_MODE="${WANDB_MODE:-offline}" \
    --env "WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-COMP767}" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env VLLM_NO_USAGE_STATS=1 \
    --env PYTHONPATH=/workspace \
    "${REPO_ROOT}/easyr1.sif" \
    bash -c "cd /workspace && python -m verl.trainer.main \
        config=${CONFIG} \
        worker.actor.model.model_path=${MODEL_PATH} \
        trainer.n_gpus_per_node=${N_GPUS} \
        ${MAX_STEPS:+trainer.max_steps=${MAX_STEPS}} \
        research.ar.start_after_steps=${WARMUP_STEPS} \
        trainer.save_checkpoint_path=/workspace/${SAVE_DIR} \
        trainer.find_last_checkpoint=true \
        worker.rollout.disable_tqdm=true \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        ${EXTRA_ARGS:-}"
