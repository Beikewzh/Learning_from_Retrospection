# Rorqual exp A — Qwen3-4B: AR + logging/checkpoints; intrinsic off in policy (see YAML).
#
# Training hyperparameters: examples/research/rorqual/baseline_ar_eta0_math12k.yaml
# This file only sets cluster/offline/runtime env.
#
# GPUs: Rorqual submit scripts use 1×4; default here matches. Override for 8-GPU nodes as needed.

export LEARS_FROZEN_YAML="${LEARS_FROZEN_YAML:-1}"
export BASE_CONFIG="${BASE_CONFIG:-examples/research/rorqual/baseline_ar_eta0_math12k.yaml}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAINER_LOGGER_JSON="${TRAINER_LOGGER_JSON:-[\"file\"]}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export USE_SINGULARITY="${USE_SINGULARITY:-1}"

export TRAINER_NNODES="${TRAINER_NNODES:-1}"
# Match examples/research/rorqual/*_fullft_8gpu.yaml (override to 2 for small interactive tests).
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

export VARIANT="${VARIANT:-lears_default}"
export RUN_ID="${RUN_ID:-rorqual_baseline_ar_eta0_n${TRAINER_NNODES:-1}g${TRAINER_N_GPUS_PER_NODE:-4}_$(date +%Y%m%d_%H%M%S)}"
export CURRICULUM_STAGE="${CURRICULUM_STAGE:-stage1_main}"
export DATASET_PRESET="${DATASET_PRESET:-math12k_math500}"

export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-COMP767}"
export WANDB_ENTITY="${WANDB_ENTITY:-zihan-wang-beike-mcgill-university}"

# export MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen__Qwen3-4B}"
