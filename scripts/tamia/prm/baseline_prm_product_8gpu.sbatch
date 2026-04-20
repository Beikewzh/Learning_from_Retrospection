#!/bin/bash
#SBATCH --job-name=tamia-prm-product-8g
#SBATCH --account=aip-tyrell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
mkdir -p slurm_logs

module load StdEnv/2023 apptainer/1.3.5

RUN_ID="tamia_baseline_prm_product_8gpu_$(date +%Y%m%d_%H%M%S)"

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
    config=examples/research/rorqual/baseline_prm_math12k.yaml \
    worker.actor.model.model_path=/workspace/models/Qwen__Qwen3-4B \
    'data.train_files=/workspace/data/math12k/train.parquet' \
    'data.val_files=/workspace/data/math500/test.parquet' \
    trainer.n_gpus_per_node=7 \
    trainer.total_epochs=1 \
    trainer.max_steps=null \
    worker.rollout.tensor_parallel_size=1 \
    research.ar.start_after_steps=375 \
    trainer.save_checkpoint_path=/workspace/checkpoints/COMP767/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    worker.rollout.disable_tqdm=true \
    trainer.logger='[\"file\"]' \
    trainer.experiment_name=${RUN_ID}"
