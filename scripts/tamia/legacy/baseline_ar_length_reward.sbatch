#!/bin/bash
#SBATCH --job-name=tamia-ar-length
#SBATCH --account=aip-dprecup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR}"
cd "${REPO}"
mkdir -p slurm_logs

module load StdEnv/2023 apptainer/1.3.5

# SLURM_JOB_ID is preserved on requeue so RUN_ID stays the same across resumes
RUN_ID="tamia_baseline_ar_length_reward_${SLURM_JOB_ID}"
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
    config=examples/research/rorqual/baseline_ar_length_reward_math12k.yaml \
    worker.actor.model.model_path=/workspace/models/Qwen__Qwen3-4B \
    'data.train_files=/workspace/data/math12k/train.parquet' \
    'data.val_files=/workspace/data/math500/test.parquet' \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    trainer.max_steps=null \
    research.ar.start_after_steps=375 \
    trainer.save_checkpoint_path=/workspace/checkpoints/COMP767/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    worker.rollout.disable_tqdm=true \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID}" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
