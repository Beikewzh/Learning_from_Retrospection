#!/bin/bash
#SBATCH --job-name=fir-math-grpo-lora-4g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_ROOT="${SCRATCH:-${HOME}/scratch}/Learning_from_Retrospection_runs"
cd "${REPO}"
mkdir -p slurm_logs "${RUN_ROOT}/checkpoints/tutorial"

module load StdEnv/2023 apptainer/1.3.5

SEED="${SEED:-1}"
RUN_LABEL="${RUN_LABEL:-final_vanilla_math_grpo_lora_4gpu_fir_seed${SEED}}"
RUN_ID="${RUN_LABEL}_job${SLURM_JOB_ID}"
PREEMPT_GRACE_SECONDS=240

auto_requeue() {
    echo "$(date): SIGUSR1 received, requesting graceful checkpoint before requeue for ${SLURM_JOB_ID}..."
    if [[ -n "${APPTAINER_PID:-}" ]] && kill -0 "${APPTAINER_PID}" 2>/dev/null; then
        kill -USR1 -- -"${APPTAINER_PID}" 2>/dev/null || true
        for ((i = 0; i < PREEMPT_GRACE_SECONDS; i++)); do
            if ! kill -0 "${APPTAINER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${APPTAINER_PID}" 2>/dev/null; then
            echo "$(date): Graceful shutdown timed out, sending SIGTERM to process group ${APPTAINER_PID}."
            kill -TERM -- -"${APPTAINER_PID}" 2>/dev/null || true
            sleep 10
        fi
    fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap auto_requeue SIGUSR1

setsid apptainer exec --nv \
  --bind "${REPO}:/workspace" \
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
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.dtype=fp16 \
    worker.rollout.seed=${SEED} \
    worker.actor.fsdp.torch_dtype=fp16 \
    worker.actor.fsdp.mp_param_dtype=fp16 \
    worker.ref.fsdp.torch_dtype=fp16 \
    worker.ref.fsdp.mp_param_dtype=fp16 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    trainer.max_steps=100 \
    trainer.val_before_train=true \
    trainer.val_freq=5 \
    trainer.save_freq=10 \
    trainer.save_limit=1 \
    trainer.save_checkpoint_path=${RUN_ROOT}/checkpoints/tutorial/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID}" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
