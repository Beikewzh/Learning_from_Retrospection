#!/bin/bash
#SBATCH --job-name=rorqual-baseline-ar
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=500G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR}"
cd "${REPO}"
mkdir -p slurm_logs

module load StdEnv/2023 apptainer/1.3.5

# Kill any stale processes left on this node by previous failed jobs.
# SLURM does not always clean up apptainer/squashfuse/ray child processes fully.
pkill -u "$(id -u)" -f "squashfuse" 2>/dev/null || true
pkill -u "$(id -u)" -f "fusermount" 2>/dev/null || true
pkill -u "$(id -u)" -f "ray" 2>/dev/null || true
sleep 5

echo "$(date): Copying model weights to \$SLURM_TMPDIR..."
mkdir -p "${SLURM_TMPDIR}/models/Qwen__Qwen3-4B"
rsync -a --info=progress2 "${REPO}/models/Qwen__Qwen3-4B/" "${SLURM_TMPDIR}/models/Qwen__Qwen3-4B/"
echo "$(date): Model copy done."

# SLURM_JOB_ID is preserved on requeue so RUN_ID stays the same across resumes
RUN_ID="rorqual_baseline_ar_logging_${SLURM_JOB_ID}"

auto_requeue() {
    echo "$(date): SIGUSR1 received — requeueing ${SLURM_JOB_ID} for checkpoint resume..."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap auto_requeue SIGUSR1

apptainer exec --nv \
  --bind "${REPO}:/workspace" \
  --bind "${SLURM_TMPDIR}/models:/localmodels" \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env WANDB_MODE=offline \
  --env WANDB_RUN_ID="${SLURM_JOB_ID}" \
  --env WANDB_RESUME=allow \
  --env VLLM_NO_USAGE_STATS=1 \
  --env RAY_raylet_start_wait_time_s=120 \
  --env PYTHONPATH=/workspace \
  "${REPO}/easyr1.sif" \
  bash -c "cd /workspace && python -m verl.trainer.main \
    config=examples/research/rorqual/baseline_ar_eta0_math12k.yaml \
    worker.actor.model.model_path=/localmodels/Qwen__Qwen3-4B \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2 \
    trainer.max_steps=null \
    research.ar.start_after_steps=375 \
    trainer.save_checkpoint_path=/workspace/checkpoints/COMP767/${RUN_ID} \
    trainer.find_last_checkpoint=true \
    worker.rollout.disable_tqdm=true \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID}" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
