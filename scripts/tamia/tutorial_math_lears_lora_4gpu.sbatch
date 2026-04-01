#!/bin/bash
#SBATCH --job-name=tamia-math-lears-lora-4g
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

RUN_ID="tamia_tutorial_math_lears_lora_4gpu_${SLURM_JOB_ID}"

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
    research.latent.source=actor_logprob \
    research.latent.layer_index=-1 \
    research.latent.include_prompt=false \
    research.latent.dtype=fp16 \
    research.latent.capture_every_n_steps=1 \
    research.buffer.backend=parquet \
    research.buffer.max_disk_gb=20.0 \
    research.buffer.max_train_samples=1024 \
    research.buffer.flush_every_n_steps=8 \
    research.ar.model_type=tiny_transformer \
    research.ar.d_model=256 \
    research.ar.n_layers=2 \
    research.ar.n_heads=4 \
    research.ar.dropout=0.1 \
    research.ar.lr=1e-4 \
    research.ar.batch_size=8 \
    research.ar.train_steps=60 \
    research.ar.train_every_n_steps=1 \
    research.ar.start_after_steps=10 \
    research.ar.min_buffer_samples=128 \
    research.ar.device=cpu \
    research.ar.max_seq_len=2048 \
    research.ar.async_enabled=false \
    research.ar.reload_every_n_steps=1 \
    research.ar.continue_from_latest=true \
    research.intrinsic.mode=weighted_additive \
    research.intrinsic.outcome_gated=true \
    research.intrinsic.success_threshold=0.5 \
    research.intrinsic.lambda_success=0.0 \
    research.intrinsic.lambda_failure=0.0 \
    research.intrinsic.normalize=per_sequence_zscore \
    research.intrinsic.normalize_scope=per_sequence_zscore \
    research.intrinsic.clip_value=3.0 \
    research.intrinsic.token_mode=dense \
    research.intrinsic.epsilon=1e-6 \
    research.intrinsic.eta=0.02 \
    research.intrinsic.gate_mode=failure_only \
    research.intrinsic.lambda_success_gate=0.25 \
    research.intrinsic.temporal_smoothing_window=3 \
    research.intrinsic.group_norm_per_timestep=false \
    research.tracing.enabled=true \
    research.tracing.flush_every_n_steps=20 \
    research.tracing.save_tokens=true \
    research.tracing.save_decoded_text=true \
    research.tracing.save_latents=true \
    research.tracing.save_ar_error=true" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
