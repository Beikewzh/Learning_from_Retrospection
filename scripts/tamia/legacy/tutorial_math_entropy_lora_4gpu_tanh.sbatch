#!/bin/bash
#SBATCH --job-name=tamia-math-entropy-tanh-4g
#SBATCH --account=aip-tyrell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/p/%u/Learning_from_Retrospection_runs/slurm_logs/slurm-%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_ROOT="${SCRATCH:-/scratch/p/${USER}}/Learning_from_Retrospection_runs"
cd "${REPO}"
mkdir -p "${RUN_ROOT}/slurm_logs" "${RUN_ROOT}/checkpoints/tutorial"

module load StdEnv/2023 apptainer/1.3.5 httpproxy

ENTROPY_ETA="${ENTROPY_ETA:-0.01}"
ENTROPY_ETA_TAG="${ENTROPY_ETA_TAG:-${ENTROPY_ETA//./p}}"
SEED="${SEED:-1}"
RUN_LABEL="${RUN_LABEL:-final_entropy_tanh_math_lora_4gpu_eta${ENTROPY_ETA_TAG}_seed${SEED}}"
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
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.dtype=fp16 \
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
    trainer.save_checkpoint_path=${RUN_ROOT}/checkpoints/tutorial/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID} \
    research.enabled=true \
    research.intrinsic.source=sampled_entropy \
    research.intrinsic.transform=tanh \
    research.intrinsic.mode=weighted_additive \
    research.intrinsic.outcome_gated=true \
    research.intrinsic.success_threshold=0.5 \
    research.intrinsic.lambda_success=0.0 \
    research.intrinsic.lambda_failure=0.0 \
    research.intrinsic.normalize=running_zscore \
    research.intrinsic.normalize_scope=running_zscore \
    research.intrinsic.clip_value=3.0 \
    research.intrinsic.token_mode=dense \
    research.intrinsic.epsilon=1e-6 \
    research.intrinsic.eta=${ENTROPY_ETA} \
    research.intrinsic.gate_mode=asymmetric \
    research.intrinsic.lambda_success_gate=1.0 \
    research.intrinsic.temporal_smoothing_window=3 \
    research.intrinsic.group_norm_per_timestep=false" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
