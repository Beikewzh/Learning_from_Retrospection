#!/bin/bash
#SBATCH --job-name=rorqual-smoke-test
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR}"
cd "${REPO}"
mkdir -p slurm_logs

module load StdEnv/2023 apptainer/1.3.5

RUN_ID="rorqual_smoke_${SLURM_JOB_ID}"
echo "Starting smoke test: RUN_ID=${RUN_ID}"

apptainer exec --nv \
  --bind "${REPO}:/workspace" \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env WANDB_MODE=offline \
  --env VLLM_NO_USAGE_STATS=1 \
  --env PYTHONPATH=/workspace \
  "${REPO}/easyr1.sif" \
  bash -c "cd /workspace && python -m verl.trainer.main \
    config=examples/research/rorqual/lears_intrinsic_math12k.yaml \
    worker.actor.model.model_path=/workspace/models/Qwen__Qwen3-4B \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    trainer.max_steps=5 \
    research.ar.start_after_steps=3 \
    trainer.save_checkpoint_path=/workspace/checkpoints/smoke/${RUN_ID} \
    trainer.find_last_checkpoint=false \
    trainer.save_freq=5 \
    worker.rollout.disable_tqdm=true \
    trainer.logger='[\"file\"]' \
    trainer.experiment_name=${RUN_ID}"

echo "Smoke test finished successfully."
