#!/bin/bash
#SBATCH --job-name=tamia-math-grpo-lora-4g-l1max
#SBATCH --account=aip-tyrell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/p/%u/Learning_from_Retrospection_runs/slurm_logs/slurm-%x-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_ROOT="${SCRATCH:-/scratch/p/${USER}}/Learning_from_Retrospection_runs"
cd "${REPO}"
mkdir -p "${RUN_ROOT}/slurm_logs"

module load StdEnv/2023 apptainer/1.3.5 httpproxy

L1_ALPHA="${L1_ALPHA:-0.01}"
L1_ALPHA_TAG="${L1_ALPHA_TAG:-${L1_ALPHA//./p}}"
TARGET_LENGTH="${TARGET_LENGTH:-512}"
DELTA="${DELTA:-0.5}"
DELTA_TAG="${DELTA_TAG:-${DELTA//./p}}"
SEED="${SEED:-1}"
RUN_LABEL="${RUN_LABEL:-final_l1max_math_grpo_lora_4gpu_a${L1_ALPHA_TAG}_t${TARGET_LENGTH}_d${DELTA_TAG}_seed${SEED}}"
RUN_ID="${RUN_LABEL}_job${SLURM_JOB_ID}"

apptainer exec --nv \
  --bind "${REPO}:/workspace" \
  --env http_proxy="${http_proxy:-}" \
  --env https_proxy="${https_proxy:-}" \
  --env HTTP_PROXY="${HTTP_PROXY:-}" \
  --env HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --env no_proxy="${no_proxy:-}" \
  --env NO_PROXY="${NO_PROXY:-}" \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env WANDB_MODE=offline \
  --env WANDB_RUN_ID="${SLURM_JOB_ID}" \
  --env WANDB_RESUME=allow \
  --env VLLM_NO_USAGE_STATS=1 \
  --env PYTHONPATH=/workspace \
  "${REPO}/easyr1.sif" \
  bash -c "cd /workspace && python -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=/workspace/models/Qwen__Qwen3-4B \
    'data.train_files=/workspace/data/math12k/train.parquet' \
    'data.val_files=/workspace/data/math500/test.parquet' \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.seed=${SEED} \
    data.max_response_length=1024 \
    data.rollout_batch_size=16 \
    data.val_batch_size=16 \
    worker.actor.global_batch_size=16 \
    worker.actor.model.lora.rank=64 \
    worker.actor.model.lora.alpha=64 \
    worker.actor.optim.lr=1e-6 \
    worker.rollout.seed=${SEED} \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.dtype=fp16 \
    worker.actor.fsdp.torch_dtype=fp16 \
    worker.actor.fsdp.mp_param_dtype=fp16 \
    worker.ref.fsdp.torch_dtype=fp16 \
    worker.ref.fsdp.mp_param_dtype=fp16 \
    worker.reward.reward_function=./examples/reward_function/math_l1_max_length.py:compute_score \
    worker.reward.reward_function_kwargs.alpha=${L1_ALPHA} \
    worker.reward.reward_function_kwargs.target_length=${TARGET_LENGTH} \
    worker.reward.reward_function_kwargs.delta=${DELTA} \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    trainer.max_steps=150 \
    trainer.val_before_train=true \
    trainer.val_freq=5 \
    trainer.save_freq=-1 \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID}"
