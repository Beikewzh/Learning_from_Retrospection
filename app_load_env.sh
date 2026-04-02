module load apptainer

export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache
export APPTAINER_TMPDIR=$SCRATCH/.apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

cd $SCRATCH/Learning_from_Retrospection
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export XDG_CACHE_HOME=$SCRATCH/.cache
export TORCH_HOME=$SCRATCH/.cache/torch
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

apptainer shell --nv --cleanenv --bind $SCRATCH/Learning_from_Retrospection:/workspace --pwd /workspace easyr1.sif
