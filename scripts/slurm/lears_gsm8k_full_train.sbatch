#!/bin/bash
#SBATCH --job-name=lears-online-full
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out

set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

join_cmd() {
    local out
    printf -v out '%q ' "$@"
    echo "${out}"
}

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-COMP767}"
WANDB_ENTITY="${WANDB_ENTITY:-zihan-wang-beike-mcgill-university}"
export WANDB_PROJECT_NAME WANDB_ENTITY

PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/slurm_logs"

# Work around known site epilog issue.
touch "${REPO_ROOT}/.no_report" || true
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    touch "${SLURM_SUBMIT_DIR}/.no_report" || true
fi

SIF_PATH="${SIF_PATH:-${REPO_ROOT}/easyr1.sif}"
CONTAINER_BIN=""
if [[ "${USE_SINGULARITY}" == "1" ]]; then
    SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${HOME}/scratch/.singularity_cache}"
    SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${HOME}/scratch/.singularity_tmp}"
    export SINGULARITY_CACHEDIR SINGULARITY_TMPDIR
    mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_BIN="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_BIN="apptainer"
    else
        if [[ -f /etc/profile ]]; then
            set +u
            # shellcheck source=/dev/null
            source /etc/profile >/dev/null 2>&1 || true
            set -u
        fi
        if command -v module >/dev/null 2>&1; then
            module load singularity/3.7.1 >/dev/null 2>&1 || true
        fi
        if command -v singularity >/dev/null 2>&1; then
            CONTAINER_BIN="singularity"
        elif command -v apptainer >/dev/null 2>&1; then
            CONTAINER_BIN="apptainer"
        fi
    fi
    if [[ -z "${CONTAINER_BIN}" ]]; then
        echo "No singularity/apptainer found. Set USE_SINGULARITY=0 to run host python." >&2
        exit 1
    fi
    if [[ ! -f "${SIF_PATH}" ]]; then
        echo "Container image not found: ${SIF_PATH}" >&2
        exit 1
    fi
fi

BASE_CONFIG="${BASE_CONFIG:-examples/research/lears_gsm8k.yaml}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/checkpoints/${WANDB_PROJECT_NAME}/lears_full}"
mkdir -p "${CACHE_ROOT}" "${SAVE_ROOT}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

to_runtime_path() {
    local path="$1"
    if [[ "${USE_SINGULARITY}" == "1" && "${path}" == "${REPO_ROOT}"* ]]; then
        printf '/workspace%s' "${path#${REPO_ROOT}}"
    else
        printf '%s' "${path}"
    fi
}

run_shell_local() {
    local cmd="$1"
    if [[ "${USE_SINGULARITY}" == "1" ]]; then
        local env_args=(
            --env "WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME}"
            --env "WANDB_ENTITY=${WANDB_ENTITY}"
            --env "HF_HOME=/workspace/.cache/huggingface"
            --env "HF_HUB_CACHE=/workspace/.cache/huggingface/hub"
            --env "HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub"
            --env "HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets"
            --env "XDG_CACHE_HOME=/workspace/.cache"
            --env "TORCH_HOME=/workspace/.cache/torch"
            --env "TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub"
            --env "PYTHONPATH=/workspace"
            --env "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-}"
            --env "WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-}"
        )
        local pass_vars=(
            WANDB_API_KEY HF_TOKEN HF_ENDPOINT
            HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
            RAY_ADDRESS RAY_USAGE_STATS_ENABLED
        )
        local var
        for var in "${pass_vars[@]}"; do
            if [[ -n "${!var:-}" ]]; then
                env_args+=(--env "${var}=${!var}")
            fi
        done
        "${CONTAINER_BIN}" exec \
            --nv \
            --cleanenv \
            --bind "${REPO_ROOT}:/workspace" \
            "${env_args[@]}" \
            "${SIF_PATH}" \
            bash -lc "cd /workspace && ${cmd}"
    else
        bash -lc "cd \"${REPO_ROOT}\" && ${cmd}"
    fi
}

run_shell_local_bg() {
    local cmd="$1"
    run_shell_local "${cmd}" &
    BG_PIDS+=("$!")
}

run_shell_on_node() {
    local node="$1"
    local cmd="$2"
    if [[ "${USE_SINGULARITY}" == "1" ]]; then
        srun --overlap --nodes=1 --ntasks=1 -w "${node}" --kill-on-bad-exit=1 \
            "${CONTAINER_BIN}" exec \
            --nv \
            --cleanenv \
            --bind "${REPO_ROOT}:/workspace" \
            --env "WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME}" \
            --env "WANDB_ENTITY=${WANDB_ENTITY}" \
            --env "HF_HOME=/workspace/.cache/huggingface" \
            --env "HF_HUB_CACHE=/workspace/.cache/huggingface/hub" \
            --env "HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub" \
            --env "HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets" \
            --env "XDG_CACHE_HOME=/workspace/.cache" \
            --env "TORCH_HOME=/workspace/.cache/torch" \
            --env "TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub" \
            --env "PYTHONPATH=/workspace" \
            --env "RAY_ADDRESS=${RAY_ADDRESS:-}" \
            --env "RAY_USAGE_STATS_ENABLED=${RAY_USAGE_STATS_ENABLED:-0}" \
            "${SIF_PATH}" \
            bash -lc "cd /workspace && ${cmd}"
    else
        srun --overlap --nodes=1 --ntasks=1 -w "${node}" --kill-on-bad-exit=1 \
            bash -lc "cd \"${REPO_ROOT}\" && ${cmd}"
    fi
}

run_shell_on_node_bg() {
    local node="$1"
    local cmd="$2"
    run_shell_on_node "${node}" "${cmd}" &
    BG_PIDS+=("$!")
}

run_python() {
    local cmd
    cmd="$(join_cmd "${PYTHON_BIN}" "$@")"
    run_shell_local "${cmd}"
}

round_product_to_int() {
    local lhs="$1"
    local rhs="$2"
    run_shell_local "${PYTHON_BIN} -c \"import math; print(max(1, int(round(float('${lhs}') * float('${rhs}')))))\""
}

compute_steps_per_epoch() {
    local script_path
    script_path="$(mktemp)"
    cat > "${script_path}" <<'PY'
import os
from omegaconf import OmegaConf

from verl.trainer.config import PPOConfig
from verl.trainer.data_loader import create_dataloader
from verl.utils.tokenizer import get_processor, get_tokenizer

cfg_path = os.environ["BASE_CONFIG_FOR_STEPS"]
overrides = [
    f"data.train_files={os.environ['TRAIN_FILES']}",
    f"data.val_files={os.environ['VAL_FILES']}",
    f"data.prompt_key={os.environ['PROMPT_KEY']}",
    f"data.answer_key={os.environ['ANSWER_KEY']}",
    f"data.format_prompt={os.environ['FORMAT_PROMPT_FOR_STEPS']}",
    f"data.max_prompt_length={os.environ['MAX_PROMPT_LENGTH']}",
    f"data.rollout_batch_size={os.environ['ROLLOUT_BATCH_SIZE']}",
    "data.mini_rollout_batch_size=null",
]

default_cfg = OmegaConf.structured(PPOConfig())
file_cfg = OmegaConf.load(cfg_path)
cli_cfg = OmegaConf.from_dotlist(overrides)
cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg)
cfg = OmegaConf.to_object(cfg)
cfg.deep_post_init()

tokenizer = get_tokenizer(
    cfg.worker.actor.model.model_path,
    override_chat_template=cfg.data.override_chat_template,
    trust_remote_code=cfg.worker.actor.model.trust_remote_code,
    use_fast=True,
)
processor = get_processor(
    cfg.worker.actor.model.model_path,
    override_chat_template=cfg.data.override_chat_template,
    trust_remote_code=cfg.worker.actor.model.trust_remote_code,
    use_fast=True,
)
train_dataloader, _ = create_dataloader(cfg.data, tokenizer, processor)
print(len(train_dataloader))
PY

    local out steps
    out="$(run_python "${script_path}")"
    rm -f "${script_path}"
    steps="$(echo "${out}" | tail -n 1 | tr -d '[:space:]')"
    if [[ ! "${steps}" =~ ^[0-9]+$ ]] || (( steps <= 0 )); then
        echo "Failed to resolve steps_per_epoch. Raw output:" >&2
        echo "${out}" >&2
        exit 1
    fi
    echo "${steps}"
}

print_startup_summary() {
    log "===== LeaRS Full Training Summary ====="
    log "Lifecycle: single trainer process (online LeaRS), no external phase restarts."
    log "AR cadence: start_after_steps=${WARMUP_STEPS}, train_every_n_steps=${ONLINE_AR_TRAIN_EVERY_N_STEPS}"
    log "Topology: allocated_nodes=${NNODES_ALLOC}, trainer.nnodes=${TRAINER_NNODES}, trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE}"
    log "Paths: REPO_ROOT=${REPO_ROOT}, SAVE_ROOT=${SAVE_ROOT}, RUN_SAVE_DIR=${RUN_SAVE_DIR}"
    log "Container: USE_SINGULARITY=${USE_SINGULARITY}, SIF_PATH=${SIF_PATH}, PYTHON_BIN=${PYTHON_BIN}"
    log "Schedule mode: ${SCHEDULE_MODE}"
    if [[ "${SCHEDULE_MODE}" == "epoch" ]]; then
        log "Epoch schedule: TOTAL_EPOCHS=${TOTAL_EPOCHS}, WARMUP_EPOCHS=${WARMUP_EPOCHS}, REFRESH_EPOCHS=${REFRESH_EPOCHS}, STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
    fi
    log "Schedule steps: TOTAL_STEPS=${TOTAL_STEPS}, WARMUP_STEPS=${WARMUP_STEPS}, REFRESH_INTERVAL=${REFRESH_INTERVAL}"
    log "Warmup guard: requested=${REQUESTED_WARMUP_STEPS}, min_required=${MIN_WARMUP_STEPS_NEEDED}, effective=${WARMUP_STEPS}"
    log "Warmup data budget: samples_per_step=${SAMPLES_PER_STEP}, warmup_sample_budget=${WARMUP_SAMPLE_BUDGET}"
    log "AR update estimate: expected_attempts=${EXPECTED_AR_ATTEMPTS} (successful updates depend on buffer sufficiency)."
    log "AR window: AR_WINDOW_INTERVALS=${AR_WINDOW_INTERVALS}, AR_WINDOW_INTERVAL_STEPS=${AR_WINDOW_INTERVAL_STEPS}"
    log "AR freshness: max_age_steps=${AR_MAX_AGE_STEPS}, stale_action=${AR_STALE_ACTION}"
    log "Variant: ${VARIANT} (lambda_success=${LAMBDA_SUCCESS}, lambda_failure=${LAMBDA_FAILURE}, ar_lr=${AR_LR}, ar_train_steps=${AR_TRAIN_STEPS})"
    log "Curriculum: stage=${CURRICULUM_STAGE}, train_files=${TRAIN_FILES}, val_files=${VAL_FILES}, prompt_key=${PROMPT_KEY}, answer_key=${ANSWER_KEY}"
    log "Data/runtime: max_prompt=${MAX_PROMPT_LENGTH}, max_response=${MAX_RESPONSE_LENGTH}, rollout_batch=${ROLLOUT_BATCH_SIZE}, actor_global_batch=${ACTOR_GLOBAL_BATCH_SIZE}, rollout_n=${ROLLOUT_N}, rollout_tp=${ROLLOUT_TP_SIZE}"
    log "AR model/runtime: device=${AR_DEVICE}, d_model=${AR_D_MODEL}, n_layers=${AR_N_LAYERS}, n_heads=${AR_N_HEADS}, dropout=${AR_DROPOUT}, batch_size=${AR_BATCH_SIZE}, min_buffer_samples=${AR_MIN_BUFFER_SAMPLES}, max_samples=${AR_MAX_SAMPLES}"
    log "W&B: project=${WANDB_PROJECT_NAME}, entity=${WANDB_ENTITY}, run_group=${RUN_ID}"
    log "W&B auth: WANDB_API_KEY=$([[ -n "${WANDB_API_KEY:-}" ]] && echo set || echo unset), WANDB_MODE=${WANDB_MODE:-online}"
    log "Ray knobs: RAY_HEAD_IP=${RAY_HEAD_IP:-auto}, RAY_PORT=${RAY_PORT:-6379}, RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}, RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
    log "DRY_RUN=${DRY_RUN}"
    log "======================================="
}

run_online_training() {
    export WANDB_RUN_GROUP="${RUN_ID}"
    export WANDB_JOB_TYPE="online_single_process"
    run_python -m verl.trainer.main \
        "${COMMON_ARGS[@]}" \
        "trainer.max_steps=${TOTAL_STEPS}" \
        "trainer.experiment_name=${RUN_ID}_online" \
        "trainer.logger=[\"file\",\"wandb\"]" \
        "trainer.project_name=${WANDB_PROJECT_NAME}" \
        "trainer.nnodes=${TRAINER_NNODES}" \
        "trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE}" \
        "trainer.save_checkpoint_path=${RUN_SAVE_DIR_RUNTIME}" \
        "trainer.find_last_checkpoint=true"
}

detect_current_step() {
    local latest
    latest="$(find "${RUN_SAVE_DIR}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)"
    if [[ -z "${latest}" ]]; then
        echo "0"
        return
    fi
    echo "${latest##*_}"
}

cleanup() {
    set +e
    if (( TRAINER_NNODES > 1 && RAY_CLUSTER_STARTED == 1 )); then
        log "Stopping Ray cluster"
        run_shell_local "$(join_cmd ray stop --force)" >/dev/null 2>&1 || true
        local node
        for node in "${WORKER_NODES[@]}"; do
            run_shell_on_node "${node}" "$(join_cmd ray stop --force)" >/dev/null 2>&1 || true
        done
    fi
    local pid
    for pid in "${BG_PIDS[@]}"; do
        kill "${pid}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

detect_alloc_gpus_per_node() {
    local raw="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
    if [[ -z "${raw}" ]]; then
        echo ""
        return
    fi
    # Handles forms like "8", "gpu:a100l:8", "8(x2)", "a100l:4(S:0-3)".
    local normalized token
    normalized="${raw%%,*}"
    normalized="${normalized%%(*}"
    if [[ "${normalized}" =~ ^[0-9]+$ ]]; then
        echo "${normalized}"
        return
    fi
    token="${normalized##*:}"
    if [[ "${token}" =~ ^[0-9]+$ ]]; then
        echo "${token}"
        return
    fi
    echo ""
}

NNODES_ALLOC="${SLURM_JOB_NUM_NODES:-1}"
if [[ "${NNODES_ALLOC}" != "1" && "${NNODES_ALLOC}" != "2" ]]; then
    echo "This script supports only 1-node or 2-node allocations. Got ${NNODES_ALLOC}." >&2
    exit 1
fi

TRAINER_NNODES="${TRAINER_NNODES:-${NNODES_ALLOC}}"
ALLOC_GPUS_PER_NODE="$(detect_alloc_gpus_per_node)"
if [[ -n "${TRAINER_N_GPUS_PER_NODE:-}" ]]; then
    : # user provided explicit value
elif [[ -n "${ALLOC_GPUS_PER_NODE}" ]]; then
    TRAINER_N_GPUS_PER_NODE="${ALLOC_GPUS_PER_NODE}"
elif [[ "${NNODES_ALLOC}" == "1" ]]; then
    TRAINER_N_GPUS_PER_NODE=8
else
    TRAINER_N_GPUS_PER_NODE=4
fi
if [[ "${TRAINER_NNODES}" != "${NNODES_ALLOC}" ]]; then
    echo "trainer.nnodes (${TRAINER_NNODES}) must match allocated nodes (${NNODES_ALLOC})." >&2
    exit 1
fi

if [[ "${TRAINER_NNODES}" != "1" && "${TRAINER_NNODES}" != "2" ]]; then
    echo "trainer.nnodes must be 1 or 2. Got ${TRAINER_NNODES}." >&2
    exit 1
fi
if [[ "${TRAINER_NNODES}" == "1" && "${TRAINER_N_GPUS_PER_NODE}" != "1" && "${TRAINER_N_GPUS_PER_NODE}" != "8" ]]; then
    echo "1-node mode supports trainer.n_gpus_per_node in {1,8}. Got ${TRAINER_N_GPUS_PER_NODE}." >&2
    exit 1
fi
if [[ "${TRAINER_NNODES}" == "2" && "${TRAINER_N_GPUS_PER_NODE}" != "4" ]]; then
    echo "2-node mode requires trainer.n_gpus_per_node=4. Got ${TRAINER_N_GPUS_PER_NODE}." >&2
    exit 1
fi

CURRICULUM_STAGE="${CURRICULUM_STAGE:-stage0_math500}"
case "${CURRICULUM_STAGE}" in
    stage0_math500)
        TRAIN_FILES="${TRAIN_FILES:-HuggingFaceH4/MATH-500@test}"
        VAL_FILES="${VAL_FILES:-HuggingFaceH4/MATH-500@test}"
        PROMPT_KEY="${PROMPT_KEY:-problem}"
        ANSWER_KEY="${ANSWER_KEY:-answer}"
        FORMAT_PROMPT="${FORMAT_PROMPT:-./examples/format_prompt/math.jinja}"
        TOTAL_EPOCHS="${TOTAL_EPOCHS:-1.0}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.25}"
        REFRESH_EPOCHS="${REFRESH_EPOCHS:-0.25}"
        AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-60}"
        AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-128}"
        ;;
    stage1_main)
        TRAIN_FILES="${TRAIN_FILES:-hiyouga/math12k@train}"
        VAL_FILES="${VAL_FILES:-hiyouga/math12k@test}"
        PROMPT_KEY="${PROMPT_KEY:-problem}"
        ANSWER_KEY="${ANSWER_KEY:-answer}"
        FORMAT_PROMPT="${FORMAT_PROMPT:-./examples/format_prompt/math.jinja}"
        TOTAL_EPOCHS="${TOTAL_EPOCHS:-2.0}"
        WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.25}"
        REFRESH_EPOCHS="${REFRESH_EPOCHS:-0.25}"
        AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-120}"
        AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-256}"
        ;;
    *)
        echo "Unsupported CURRICULUM_STAGE=${CURRICULUM_STAGE}. Use one of: stage0_math500, stage1_main." >&2
        exit 1
        ;;
esac

VARIANT="${VARIANT:-lears_default}"
case "${VARIANT}" in
    lears_default)
        LAMBDA_SUCCESS="${LAMBDA_SUCCESS:-0.1}"
        LAMBDA_FAILURE="${LAMBDA_FAILURE:-0.1}"
        AR_LR="${AR_LR:-1e-4}"
        AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-120}"
        ;;
    lears_weak)
        LAMBDA_SUCCESS="${LAMBDA_SUCCESS:-0.05}"
        LAMBDA_FAILURE="${LAMBDA_FAILURE:-0.05}"
        AR_LR="${AR_LR:-1e-4}"
        AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-120}"
        ;;
    lears_asym)
        LAMBDA_SUCCESS="${LAMBDA_SUCCESS:-0.05}"
        LAMBDA_FAILURE="${LAMBDA_FAILURE:-0.2}"
        AR_LR="${AR_LR:-1e-4}"
        AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-120}"
        ;;
    lears_fastar)
        LAMBDA_SUCCESS="${LAMBDA_SUCCESS:-0.1}"
        LAMBDA_FAILURE="${LAMBDA_FAILURE:-0.1}"
        AR_LR="${AR_LR:-3e-4}"
        AR_TRAIN_STEPS="${AR_TRAIN_STEPS:-200}"
        ;;
    *)
        echo "Unsupported VARIANT=${VARIANT}. Use one of: lears_default, lears_weak, lears_asym, lears_fastar." >&2
        exit 1
        ;;
esac

SCHEDULE_MODE="${SCHEDULE_MODE:-epoch}"
if [[ "${SCHEDULE_MODE}" != "epoch" && "${SCHEDULE_MODE}" != "step" ]]; then
    echo "Unsupported SCHEDULE_MODE=${SCHEDULE_MODE}. Use one of: epoch, step." >&2
    exit 1
fi

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-2}"
if [[ -n "${ROLLOUT_TP_SIZE:-}" ]]; then
    ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE}"
elif [[ "${TRAINER_N_GPUS_PER_NODE}" == "1" ]]; then
    ROLLOUT_TP_SIZE=1
else
    ROLLOUT_TP_SIZE=2
fi
SAVE_FREQ="${SAVE_FREQ:-50}"

AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-2}"
AR_N_HEADS="${AR_N_HEADS:-4}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-8}"
AR_MIN_BUFFER_SAMPLES="${AR_MIN_BUFFER_SAMPLES:-128}"
AR_MAX_SAMPLES="${AR_MAX_SAMPLES:-100000}"
AR_DEVICE="${AR_DEVICE:-cpu}"
if [[ "${AR_DEVICE}" != "cpu" && "${AR_DEVICE}" != "cuda" ]]; then
    echo "AR_DEVICE must be one of: cpu, cuda." >&2
    exit 1
fi

BASE_CONFIG_RUNTIME="$(to_runtime_path "${BASE_CONFIG}")"
FORMAT_PROMPT_RUNTIME="$(to_runtime_path "${FORMAT_PROMPT}")"

if [[ "${SCHEDULE_MODE}" == "epoch" ]]; then
    run_shell_local "${PYTHON_BIN} -c \"vals=[float('${TOTAL_EPOCHS}'), float('${WARMUP_EPOCHS}'), float('${REFRESH_EPOCHS}')]; import sys; sys.exit(0 if all(v > 0 for v in vals) else 1)\"" \
        || {
            echo "TOTAL_EPOCHS, WARMUP_EPOCHS, and REFRESH_EPOCHS must all be > 0 in epoch mode." >&2
            exit 1
        }

    export BASE_CONFIG_FOR_STEPS="${BASE_CONFIG_RUNTIME}"
    export TRAIN_FILES VAL_FILES PROMPT_KEY ANSWER_KEY
    export FORMAT_PROMPT_FOR_STEPS="${FORMAT_PROMPT_RUNTIME}"
    export MAX_PROMPT_LENGTH ROLLOUT_BATCH_SIZE
    STEPS_PER_EPOCH="$(compute_steps_per_epoch)"
    TOTAL_STEPS="$(round_product_to_int "${TOTAL_EPOCHS}" "${STEPS_PER_EPOCH}" | tail -n 1 | tr -d '[:space:]')"
    REQUESTED_WARMUP_STEPS="$(round_product_to_int "${WARMUP_EPOCHS}" "${STEPS_PER_EPOCH}" | tail -n 1 | tr -d '[:space:]')"
    WARMUP_STEPS="${REQUESTED_WARMUP_STEPS}"
    REFRESH_INTERVAL="$(round_product_to_int "${REFRESH_EPOCHS}" "${STEPS_PER_EPOCH}" | tail -n 1 | tr -d '[:space:]')"
else
    STEPS_PER_EPOCH=0
    TOTAL_STEPS="${TOTAL_STEPS:-1868}"
    REQUESTED_WARMUP_STEPS="${WARMUP_STEPS:-256}"
    WARMUP_STEPS="${REQUESTED_WARMUP_STEPS}"
    REFRESH_INTERVAL="${REFRESH_INTERVAL:-256}"
fi

AR_WINDOW_INTERVALS="${AR_WINDOW_INTERVALS:-2}"
AR_WINDOW_INTERVAL_STEPS="${AR_WINDOW_INTERVAL_STEPS:-${REFRESH_INTERVAL}}"
AR_MAX_AGE_STEPS="${AR_MAX_AGE_STEPS:-${REFRESH_INTERVAL}}"
AR_STALE_ACTION="${AR_STALE_ACTION:-warn}"
ONLINE_AR_TRAIN_EVERY_N_STEPS="${ONLINE_AR_TRAIN_EVERY_N_STEPS:-${REFRESH_INTERVAL}}"

if (( TOTAL_STEPS <= 0 || WARMUP_STEPS <= 0 || REFRESH_INTERVAL <= 0 )); then
    echo "TOTAL_STEPS, WARMUP_STEPS, and REFRESH_INTERVAL must all be > 0." >&2
    exit 1
fi
if (( AR_WINDOW_INTERVALS < 0 || AR_WINDOW_INTERVAL_STEPS <= 0 )); then
    echo "AR_WINDOW_INTERVALS must be >= 0 and AR_WINDOW_INTERVAL_STEPS must be > 0." >&2
    exit 1
fi
if (( AR_MAX_AGE_STEPS <= 0 )); then
    echo "AR_MAX_AGE_STEPS must be > 0." >&2
    exit 1
fi
if [[ "${AR_STALE_ACTION}" != "warn" && "${AR_STALE_ACTION}" != "fail" ]]; then
    echo "AR_STALE_ACTION must be one of: warn, fail." >&2
    exit 1
fi
if (( ONLINE_AR_TRAIN_EVERY_N_STEPS <= 0 )); then
    echo "ONLINE_AR_TRAIN_EVERY_N_STEPS must be > 0." >&2
    exit 1
fi

SAMPLES_PER_STEP=$((ROLLOUT_BATCH_SIZE * ROLLOUT_N))
WARMUP_SAMPLE_BUDGET=0
EXPECTED_AR_ATTEMPTS=0
MIN_WARMUP_STEPS_NEEDED=$(((AR_MIN_BUFFER_SAMPLES + SAMPLES_PER_STEP - 1) / SAMPLES_PER_STEP))
if (( WARMUP_STEPS < MIN_WARMUP_STEPS_NEEDED )); then
    WARMUP_STEPS="${MIN_WARMUP_STEPS_NEEDED}"
fi
if (( WARMUP_STEPS > TOTAL_STEPS )); then
    TOTAL_STEPS="${WARMUP_STEPS}"
fi
WARMUP_SAMPLE_BUDGET=$((WARMUP_STEPS * SAMPLES_PER_STEP))
if (( TOTAL_STEPS >= WARMUP_STEPS )); then
    EXPECTED_AR_ATTEMPTS=$((((TOTAL_STEPS - WARMUP_STEPS) / ONLINE_AR_TRAIN_EVERY_N_STEPS) + 1))
fi

RUN_ID="${RUN_ID:-${VARIANT}_n${TRAINER_NNODES}g${TRAINER_N_GPUS_PER_NODE}_$(date +%Y%m%d_%H%M%S)}"
RUN_SAVE_DIR="${SAVE_ROOT}/${RUN_ID}"
mkdir -p "${RUN_SAVE_DIR}"
RUN_SAVE_DIR_RUNTIME="$(to_runtime_path "${RUN_SAVE_DIR}")"

COMMON_ARGS=(
    "config=${BASE_CONFIG_RUNTIME}"
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.prompt_key=${PROMPT_KEY}"
    "data.answer_key=${ANSWER_KEY}"
    "data.format_prompt=${FORMAT_PROMPT_RUNTIME}"
    "research.enabled=true"
    "research.latent.include_prompt=false"
    "research.buffer.max_train_samples=${AR_MAX_SAMPLES}"
    "research.ar.device=${AR_DEVICE}"
    "research.ar.max_seq_len=${MAX_RESPONSE_LENGTH}"
    "research.ar.d_model=${AR_D_MODEL}"
    "research.ar.n_layers=${AR_N_LAYERS}"
    "research.ar.n_heads=${AR_N_HEADS}"
    "research.ar.dropout=${AR_DROPOUT}"
    "research.ar.batch_size=${AR_BATCH_SIZE}"
    "research.ar.lr=${AR_LR}"
    "research.ar.train_steps=${AR_TRAIN_STEPS}"
    "research.ar.min_buffer_samples=${AR_MIN_BUFFER_SAMPLES}"
    "research.ar.start_after_steps=${WARMUP_STEPS}"
    "research.ar.train_every_n_steps=${ONLINE_AR_TRAIN_EVERY_N_STEPS}"
    "research.ar.window_intervals=${AR_WINDOW_INTERVALS}"
    "research.ar.window_interval_steps=${AR_WINDOW_INTERVAL_STEPS}"
    "research.ar.max_age_steps=${AR_MAX_AGE_STEPS}"
    "research.ar.stale_action=${AR_STALE_ACTION}"
    "research.intrinsic.lambda_success=${LAMBDA_SUCCESS}"
    "research.intrinsic.lambda_failure=${LAMBDA_FAILURE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
    "worker.actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE}"
    "worker.rollout.n=${ROLLOUT_N}"
    "worker.rollout.tensor_parallel_size=${ROLLOUT_TP_SIZE}"
    "worker.rollout.val_override_config.n=1"
    "worker.rollout.disable_tqdm=true"
    "worker.actor.padding_free=false"
    "trainer.val_freq=-1"
    "trainer.val_before_train=false"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.save_limit=3"
)

BG_PIDS=()
WORKER_NODES=()
RAY_CLUSTER_STARTED=0

print_startup_summary

if (( TRAINER_NNODES > 1 )); then
    mapfile -t NODE_ARRAY < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    LOCAL_HOST="$(hostname -s)"
    RAY_HEAD_IP="${RAY_HEAD_IP:-}"
    if [[ -n "${RAY_HEAD_IP}" ]]; then
        HEAD_IP="${RAY_HEAD_IP}"
    else
        HEAD_IP="$(hostname -I | awk '{print $1}')"
    fi
    RAY_PORT="${RAY_PORT:-6379}"
    RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
    RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
    export RAY_USAGE_STATS_ENABLED=0
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

    node_name=""
    for node_name in "${NODE_ARRAY[@]}"; do
        if [[ "${node_name}" == "${LOCAL_HOST}" ]]; then
            continue
        fi
        WORKER_NODES+=("${node_name}")
    done
    if (( ${#WORKER_NODES[@]} == 0 )); then
        echo "nnodes=2 requested but no worker node discovered (SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset})." >&2
        exit 1
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
        local_head_cmd="$(join_cmd ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --disable-usage-stats --num-cpus="${RAY_NUM_CPUS}" --num-gpus="${TRAINER_N_GPUS_PER_NODE}" --block)"
        log "Starting Ray head on ${LOCAL_HOST} at ${RAY_ADDRESS}"
        run_shell_local_bg "${local_head_cmd}"
        RAY_CLUSTER_STARTED=1
        sleep 10

        for node_name in "${WORKER_NODES[@]}"; do
            worker_cmd="$(join_cmd ray start --address="${RAY_ADDRESS}" --disable-usage-stats --num-cpus="${RAY_NUM_CPUS}" --num-gpus="${TRAINER_N_GPUS_PER_NODE}" --block)"
            log "Starting Ray worker on ${node_name}"
            run_shell_on_node_bg "${node_name}" "${worker_cmd}"
            sleep 4
        done
        sleep 10
        run_shell_local "$(join_cmd ray status --address="${RAY_ADDRESS}")"
    fi
fi

current_step="$(detect_current_step)"
log "Detected current_step=${current_step}"
if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1 set; exiting before launching training."
    exit 0
fi

if (( current_step >= TOTAL_STEPS )); then
    log "Run already reached TOTAL_STEPS=${TOTAL_STEPS}. Nothing to do."
    exit 0
fi

log "Launching online LeaRS training: current_step=${current_step}, target=${TOTAL_STEPS}"
run_online_training
current_step="$(detect_current_step)"
log "Online training completed at current_step=${current_step}"
