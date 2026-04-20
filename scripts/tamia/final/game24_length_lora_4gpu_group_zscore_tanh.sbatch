#!/bin/bash
#SBATCH --job-name=tamia-24game-length-groupz-tanh-4g
#SBATCH --account=aip-tyrell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/p/%u/Learning_from_Retrospection_runs/slurm_logs/slurm-%x-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_ROOT="${SCRATCH:-/scratch/p/${USER}}/Learning_from_Retrospection_runs"
cd "${REPO}"
mkdir -p "${RUN_ROOT}/slurm_logs" "${RUN_ROOT}/checkpoints/tutorial"

module load StdEnv/2023 apptainer/1.3.5 httpproxy

LENGTH_ETA="${LENGTH_ETA:-0.01}"
LENGTH_ETA_TAG="${LENGTH_ETA_TAG:-${LENGTH_ETA//./p}}"
SEED="${SEED:-1}"
RUN_LABEL="${RUN_LABEL:-game24_length_groupz_tanh_lora_4gpu_eta${LENGTH_ETA_TAG}_seed${SEED}}"
RUN_ID="${RUN_LABEL}_job${SLURM_JOB_ID}"

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
    'data.train_files=/workspace/vendor/24game/data/24game_grpo/train.parquet' \
    'data.val_files=/workspace/vendor/24game/data/24game_grpo/val.parquet' \
    data.prompt_key=question \
    data.answer_key=reward_model \
    data.format_prompt=./examples/format_prompt/game24.jinja \
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
    worker.reward.reward_function=./examples/reward_function/game24.py:compute_score \
    worker.reward.reward_function_kwargs.format_weight=0.1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    trainer.max_steps=100 \
    trainer.val_before_train=true \
    trainer.val_freq=5 \
    trainer.save_freq=-1 \
    trainer.logger='[\"file\",\"wandb\"]' \
    trainer.experiment_name=${RUN_ID} \
    research.enabled=true \
    research.intrinsic.source=response_length \
    research.intrinsic.transform=tanh \
    research.intrinsic.mode=weighted_additive \
    research.intrinsic.outcome_gated=true \
    research.intrinsic.success_threshold=0.5 \
    research.intrinsic.lambda_success=0.0 \
    research.intrinsic.lambda_failure=0.0 \
    research.intrinsic.normalize=none \
    research.intrinsic.normalize_scope=none \
    research.intrinsic.clip_value=3.0 \
    research.intrinsic.token_mode=dense \
    research.intrinsic.epsilon=1e-6 \
    research.intrinsic.eta=${LENGTH_ETA} \
    research.intrinsic.gate_mode=asymmetric \
    research.intrinsic.lambda_success_gate=1.0 \
    research.intrinsic.temporal_smoothing_window=3 \
    research.intrinsic.group_norm_per_timestep=true \
    research.intrinsic.group_norm_scope=pooled_sequence_group" &

APPTAINER_PID=$!
wait "${APPTAINER_PID}" || exit $?
