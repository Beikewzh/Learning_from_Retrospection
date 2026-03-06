# EasyR1 on HPC (no Docker): use Singularity / Apptainer

On Mila HPC you have **Singularity** (same as Apptainer). Use it like this.

## If you get "disk quota exceeded" when pulling

Singularity uses your **home directory** for cache by default; home often has a small quota. Use **scratch** for cache and temp:

```bash
export SINGULARITY_CACHEDIR=/home/mila/z/zihan.wang/scratch/.singularity_cache
export SINGULARITY_TMPDIR=/home/mila/z/zihan.wang/scratch/.singularity_tmp
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR
```

Then run the pull (and keep these set when you run `singularity shell` / `exec` later).

## 1. Load Singularity and pull the image

```bash
module load singularity/3.7.1

# Use scratch for cache/tmp to avoid home quota (see above)
export SINGULARITY_CACHEDIR=/home/mila/z/zihan.wang/scratch/.singularity_cache
export SINGULARITY_TMPDIR=/home/mila/z/zihan.wang/scratch/.singularity_tmp
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

cd /home/mila/z/zihan.wang/scratch/Learning_from_Retrospection
singularity pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

## 2. Run a shell inside the container (GPU + bind your dir)

Bind your scratch so you can see the repo and data inside the container. Use your real path:

```bash
singularity shell --nv --cleanenv --bind /home/mila/z/zihan.wang/scratch/Learning_from_Retrospection:/workspace easyr1.sif
cd /workspace
```


Inside the container, your project is at `/workspace`. Run training from there.

## 3. Or run a command directly

```bash
singularity exec --nv --cleanenv --bind /home/mila/z/zihan.wang/scratch/Learning_from_Retrospection:/workspace easyr1.sif bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

- `--nv`: use NVIDIA GPUs  
- `--cleanenv`: don’t inherit host env (avoids conflicts)  
- `--bind LOCAL:CONTAINER`: mount your dir (use your paths)

If your cluster uses `apptainer` instead of `singularity`, replace `singularity` with `apptainer` in the commands above.
