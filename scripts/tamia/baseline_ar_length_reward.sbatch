#!/bin/bash
#SBATCH --job-name=tamia-ar-length
#SBATCH --account=aip-dprecup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
mkdir -p slurm_logs

module load StdEnv/2023 apptainer/1.3.5

RUN_ID="rorqual_baseline_ar_length_reward_$(date +%Y%m%d_%H%M%S)"

exec apptainer exec --nv \
  --bind "${REPO}:/workspace" \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env WANDB_MODE=offline \
  --env VLLM_NO_USAGE_STATS=1 \
  --env PYTHONPATH=/workspace \
  "${REPO}/easyr1.sif" \
  bash -c "cd /workspace && python -m verl.trainer.main \
    config=examples/research/rorqual/baseline_ar_length_reward_math12k.yaml \
    worker.actor.model.model_path=/workspace/models/Qwen__Qwen3-4B \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    trainer.max_steps=null \
    research.ar.start_after_steps=375 \
    trainer.save_checkpoint_path=/workspace/checkpoints/COMP767/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    worker.rollout.disable_tqdm=true \
    trainer.logger='[\"file\"]' \
    trainer.experiment_name=${RUN_ID}"
