#!/bin/bash
#SBATCH --job-name=qwen3_4b_math300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/p/psli/Learning_from_Retrospection}"
SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"

# Replace these with local paths visible on the compute node.
MODEL_PATH="${MODEL_PATH:-/path/to/local/Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/local/math12k_train.jsonl}"
VAL_FILES="${VAL_FILES:-/path/to/local/math500_test.jsonl}"

cd "${REPO_ROOT}"
mkdir -p slurm_logs .cache checkpoints

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp}"

srun apptainer exec --nv \
  --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
  --env HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
  --env XDG_CACHE_HOME=/workspace/.cache \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env PYTHONPATH=/workspace \
  --env WANDB_MODE=offline \
  "${SIF_PATH}" \
  bash -lc "
    cd /workspace
    python3 -m verl.trainer.main \
      config=examples/research/lears_math12k_math500.yaml \
      research.enabled=false \
      worker.actor.model.model_path=${MODEL_PATH} \
      data.train_files=${TRAIN_FILES} \
      data.val_files=${VAL_FILES} \
      data.prompt_key=problem \
      data.answer_key=answer \
      data.format_prompt=./examples/format_prompt/math.jinja \
      data.max_prompt_length=512 \
      data.max_response_length=4096 \
      data.rollout_batch_size=8 \
      data.val_batch_size=16 \
      worker.actor.global_batch_size=8 \
      worker.rollout.n=2 \
      worker.rollout.tensor_parallel_size=2 \
      worker.rollout.gpu_memory_utilization=0.6 \
      worker.rollout.disable_tqdm=true \
      worker.rollout.val_override_config.n=1 \
      worker.rollout.val_override_config.temperature=0.0 \
      worker.rollout.val_override_config.top_p=1.0 \
      trainer.project_name=baseline_rl \
      trainer.experiment_name=qwen3_4b_math12k_math500_300steps \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=8 \
      trainer.max_steps=300 \
      trainer.val_before_train=true \
      trainer.val_freq=25 \
      trainer.save_freq=100 \
      trainer.save_limit=2 \
      trainer.val_generations_to_log=0 \
      'trainer.logger=[\"file\"]'
  "
