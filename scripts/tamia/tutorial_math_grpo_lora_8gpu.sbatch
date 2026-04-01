#!/bin/bash
#SBATCH --job-name=tamia-math-grpo-lora-8g
#SBATCH --account=aip-tyrell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/p/%u/Learning_from_Retrospection_runs/slurm_logs/slurm-%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_ROOT="${SCRATCH:-/scratch/p/${USER}}/Learning_from_Retrospection_runs"
cd "${REPO}"
mkdir -p "${RUN_ROOT}/slurm_logs" "${RUN_ROOT}/checkpoints/tutorial"

module load StdEnv/2023 apptainer/1.3.5 httpproxy

RUN_ID="tamia_tutorial_math_grpo_lora_8gpu_${SLURM_JOB_ID}"

auto_requeue() {
    echo "$(date): SIGUSR1 received, requeueing ${SLURM_JOB_ID} for checkpoint resume..."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap auto_requeue SIGUSR1

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
    data.max_response_length=1024 \
    data.rollout_batch_size=16 \
    data.val_batch_size=16 \
    worker.actor.global_batch_size=16 \
    worker.actor.model.lora.rank=64 \
    worker.actor.model.lora.alpha=64 \
    worker.actor.optim.lr=1e-6 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.dtype=fp16 \
    worker.rollout.n=4 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.actor.fsdp.torch_dtype=fp16 \
    worker.actor.fsdp.mp_param_dtype=fp16 \
    worker.ref.fsdp.torch_dtype=fp16 \
    worker.ref.fsdp.mp_param_dtype=fp16 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.max_steps=300 \
    trainer.val_before_train=true \
    trainer.val_freq=25 \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path=${RUN_ROOT}/checkpoints/tutorial/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID}" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
