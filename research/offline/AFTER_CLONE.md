# Offline Latent Analysis After Clone

This note explains the offline latent-analysis additions that sit alongside the original online LeaRS RL code.

The repo now has a separate offline workflow under `research/` for:

1. exporting or normalizing a reasoning dataset,
2. collecting latent trajectories from pretrained / post-trained models,
3. storing those trajectories in a Parquet latent buffer,
4. analyzing success vs failure offline in notebooks.

This does not change the online RL training path under `verl/`.

## What Was Added

### Offline collection scripts

- `research/scripts/predownload_hf_assets.py`
  - pre-downloads models and optional datasets into a scratch-backed cache
- `research/scripts/export_math500_jsonl.py`
  - exports `HuggingFaceH4/MATH-500` into the local JSONL format used by the collector
- `research/scripts/prepare_reasoning_jsonl.py`
  - normalizes an existing JSONL into `{"id","question","answer"}`
- `research/scripts/collect_latents_offline.py`
  - runs a model, generates responses, extracts response-token latents, and stores them
- `research/scripts/inspect_buffer.py`
  - prints buffer contents and basic stats
- `research/scripts/train_ar_offline.py`
  - trains the existing AR model on collected offline latents

### Offline analysis notebooks

- `research/notebooks/offline_latent_pca_umap.ipynb`
  - PCA / UMAP analysis for one collected run
- `research/notebooks/compare_models_same_questions.ipynb`
  - aligns two runs on the same question IDs and compares models side by side

### Slurm launchers

- `scripts/slurm/offline_collect_math500_single_model.sbatch`
  - one model per job, resumable
- `scripts/slurm/submit_offline_collect_math500_multi_model.sh`
  - submits one single-model job per model so Slurm can parallelize them

## Scratch Cache

The offline workflow assumes Hugging Face cache should live under the repo's scratch-backed `.cache/`.

Important container-path rule:

- Slurm jobs run the repo inside the container at `/workspace`, not at the host scratch path.
- Host paths like `/network/scratch/.../Learning_from_Retrospection/...` can be passed to wrappers on the host, but inside Python scripts they must resolve to `/workspace/...`.
- When adding new scripts, do not blindly call `.resolve()` on host repo paths inside the container. Use the existing repo-path remapping pattern instead.
- If a job fails trying to create paths under `/network/...` from inside the container, that is a host/container path bug, not a cache bug.

Use:

```bash
--cache-root .cache
```

or let the Slurm scripts set:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `XDG_CACHE_HOME`
- `TORCH_HOME`

under `.cache/`.

## Basic Usage

### 1. Pre-download model and dataset

Example:

```bash
python3 research/scripts/predownload_hf_assets.py \
  --model Qwen/Qwen3-4B \
  --dataset HuggingFaceH4/MATH-500 \
  --dataset-split test \
  --cache-root .cache
```

### 2. Export MATH-500 locally

Example:

```bash
python3 research/scripts/export_math500_jsonl.py \
  --cache-root .cache \
  --limit 500
```

This creates a local JSONL file used by the collector.

### 3. Collect offline latent trajectories

Example:

```bash
python3 -m research.scripts.collect_latents_offline \
  --model Qwen/Qwen3-4B \
  --input data/math500_local.jsonl \
  --output-dir outputs/qwen3_4b_math500_latents \
  --cache-root .cache \
  --prompt-template examples/format_prompt/math.jinja \
  --layer-index -1 \
  --max-prompt-length 512 \
  --max-response-length 4096 \
  --temperature 0.0 \
  --limit 20 \
  --overwrite
```

### 4. Collect multiple samples per question

The collector now supports repeated sampling per question:

```bash
python3 -m research.scripts.collect_latents_offline \
  --model Qwen/Qwen3-4B \
  --input data/math500_local.jsonl \
  --output-dir outputs/qwen3_4b_math500_temp1_k32 \
  --cache-root .cache \
  --prompt-template examples/format_prompt/math.jinja \
  --temperature 1.0 \
  --num-samples-per-question 32 \
  --seed 1 \
  --limit 100 \
  --resume
```

Important metadata fields for the multi-sample path:

- `uid`
  - unique per collected sample, e.g. `question_uid::sample_00`
- `question_uid`
  - original question identifier
- `sample_idx`
  - sample number for that question
- `generation_seed`
  - actual seed used for that sampled response

### 5. Inspect the saved buffer

```bash
python3 -m research.scripts.inspect_buffer \
  --buffer-dir outputs/qwen3_4b_math500_temp1_k32/buffer \
  --show 5
```

### 6. Open the notebooks

Run these in the same environment where the collector works, typically the `easyr1.sif` image:

- `research/notebooks/offline_latent_pca_umap.ipynb`
- `research/notebooks/compare_models_same_questions.ipynb`

## Parallel Slurm Usage

Preferred path:

```bash
LIMIT=100 NUM_SAMPLES_PER_QUESTION=32 TEMPERATURE=1.0 MAX_RESPONSE_LENGTH=4096 \
bash scripts/slurm/submit_offline_collect_math500_multi_model.sh
```

This submits:

- one job for `Qwen/Qwen3-4B`
- one job for `allenai/Llama-3.1-Tulu-3-8B`

Each job writes to its own model-specific output folder.

## Preemption / Resume

The single-model Slurm launcher is set up for preemptible collection jobs.

It now:

- traps Slurm preemption signals,
- requeues itself,
- reruns the collector with `--resume`,
- avoids overwriting previous progress by default.

Collector-side resume works by:

- reading existing `metadata.jsonl`,
- skipping already collected sample IDs,
- continuing from unfinished sample IDs.

Safe defaults in the single-model Slurm launcher:

- `RESUME=1`
- `OVERWRITE=0`
- `FLUSH_EVERY_N_SAMPLES=1`

If you truly want a fresh run:

```bash
OVERWRITE=1 RESUME=0 sbatch ...
```

## What Should Be In `.gitignore`

The following generated / local artifacts should stay untracked:

- `.cache/`
- `outputs/`
- `checkpoints/`
- `wandb/`
- `slurm_logs/`
- generated local dataset exports such as:
  - `data/math500_local*.jsonl`
  - `data/*_normalized.jsonl`

Already ignored in this repo:

- `.cache`
- `outputs/`
- `checkpoints/`
- `wandb/`
- `slurm_logs/`
- `proposal/`

The new offline workflow also added:

- `data/math500_local*.jsonl`
- `data/*_normalized.jsonl`

to `.gitignore`.

## Recommended First Checks After Clone

1. verify the container path:
   - `easyr1.sif`
2. verify cache goes to `.cache/`, not `~/.cache/`
3. run a tiny collection smoke test:

```bash
LIMIT=5 NUM_SAMPLES_PER_QUESTION=2 TEMPERATURE=1.0 MAX_RESPONSE_LENGTH=1024 \
bash scripts/slurm/submit_offline_collect_math500_multi_model.sh
```

4. inspect:
   - `metadata.jsonl`
   - `collection_summary.json`
   - `buffer/manifest.json`

That is enough to confirm the offline path is wired correctly before scaling up.
