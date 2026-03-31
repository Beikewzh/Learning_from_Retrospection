# Offline Analysis Pipeline

This guide explains the offline latent-analysis workflow end to end for someone new to the repo.

It covers:

1. what the offline pipeline does,
2. which artifact is the canonical output,
3. how to run one model cleanly,
4. how to run optional held-out split evaluation,
5. where to point the notebooks.

The offline workflow is separate from online RL training under `verl.trainer.main`.

## What The Pipeline Does

For one model, the canonical offline path is:

1. export `MATH-500` once to local JSONL,
2. predownload the model once,
3. collect latent trajectories in parallel shards,
4. merge the collection shards,
5. train one offline AR model on the merged run,
6. compute per-trajectory AR + spectrum metrics in parallel shards,
7. merge those metrics into one file for notebook analysis.

Optional:

8. run question-level held-out AR split evaluation.

## Canonical Artifact

The main output of the offline pipeline is:

```text
<run>/merged/analysis_parallel/merged/metrics.jsonl
```

Example:

```text
outputs/offline_math500_temp1_k32/qwen_qwen3_4b_limit500/merged/analysis_parallel/merged/metrics.jsonl
```

This is the default notebook input for the main AR + spectrum analysis.

The older spectrum-only branch still exists, but it is optional and not the default path.

## Directory Layout

For a model tag like `qwen_qwen3_4b`, the main output tree looks like:

```text
outputs/offline_math500_temp1_k32/qwen_qwen3_4b_limit500/
  shard_00/
  shard_01/
  ...
  merged/
    buffer/
    metadata.jsonl
    collection_summary.json
    analysis_parallel/
      ar_model/
      shards/
      merged/
        metrics.jsonl
        summary.json
    spectrum_parallel/            # optional, only if explicitly enabled
    ar_split_eval_parallel/       # optional, only if explicitly run
```

## Main Entry Points

Main shell wrapper:

- [submit_offline_model_end_to_end.sh](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/scripts/slurm/submit_offline_model_end_to_end.sh)

Main Python scripts:

- [collect_latents_offline.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/collect_latents_offline.py)
- [train_ar_on_run.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/train_ar_on_run.py)
- [compute_ar_spectrum_metrics.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/compute_ar_spectrum_metrics.py)
- [merge_analysis_metrics.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/merge_analysis_metrics.py)
- [eval_ar_question_splits.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/eval_ar_question_splits.py)
- [merge_split_eval_results.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/merge_split_eval_results.py)

Shared helpers:

- [offline_common.sh](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/scripts/slurm/offline_common.sh)
- [offline_utils.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/offline_utils.py)

## One Clean End-To-End Run

From repo root:

```bash
cd /network/scratch/p/pingsheng.li/Learning_from_Retrospection
```

Run one model:

```bash
MODEL_ID="Qwen/Qwen3-4B" \
MODEL_TAG="qwen_qwen3_4b" \
NUM_SHARDS=10 \
NUM_SAMPLES_PER_QUESTION=32 \
LIMIT=500 \
TEMPERATURE=1.0 \
AR_DEVICE=cpu \
AR_MAX_SAMPLES=20000 \
AR_TRAIN_STEPS=5000 \
AR_BATCH_SIZE=8 \
SPILL_TRAIN_SEQUENCES=1 \
KEEP_SPILLED_SEQUENCES=0 \
bash scripts/slurm/submit_offline_model_end_to_end.sh
```

What this submits:

1. one predownload/export job,
2. `NUM_SHARDS` collection jobs,
3. one collection merge job,
4. one AR train job,
5. `NUM_SHARDS` AR+spectrum metric jobs,
6. one merged metrics job.

Default behavior:

- canonical AR+spectrum branch runs,
- spectrum-only branch does not run unless requested.

If you also want the legacy spectrum-only branch:

```bash
RUN_SPECTRUM_ONLY_ANALYSIS=1 \
MODEL_ID="Qwen/Qwen3-4B" \
MODEL_TAG="qwen_qwen3_4b" \
bash scripts/slurm/submit_offline_model_end_to_end.sh
```

## Key Environment Variables

These are the ones most people actually need:

- `MODEL_ID`
- `MODEL_TAG`
- `NUM_SHARDS`
- `NUM_SAMPLES_PER_QUESTION`
- `LIMIT`
- `TEMPERATURE`
- `AR_DEVICE`
- `AR_MAX_SAMPLES`
- `AR_TRAIN_STEPS`
- `AR_BATCH_SIZE`
- `SPILL_TRAIN_SEQUENCES`

Usually you do not need to override:

- `CACHE_ROOT`
- `OUTPUT_BASE`
- `USE_REASONING_SPAN`

## Optional Held-Out Split Evaluation

This path is separate from the canonical merged metrics path.

It runs non-overlapping question-level folds for AR evaluation.

Wrapper:

- [submit_offline_eval_ar_question_splits.sh](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/scripts/slurm/submit_offline_eval_ar_question_splits.sh)

Example:

```bash
RUN_DIR=/network/scratch/p/pingsheng.li/Learning_from_Retrospection/outputs/offline_math500_temp1_k32/qwen_qwen3_4b_limit500/merged \
NUM_SPLITS=5 \
USE_REASONING_SPAN=1 \
AR_DEVICE=cpu \
AR_MAX_SAMPLES=10000 \
AR_TRAIN_STEPS=3750 \
AR_BATCH_SIZE=8 \
SPILL_TRAIN_SEQUENCES=1 \
KEEP_SPILLED_SEQUENCES=0 \
bash scripts/slurm/submit_offline_eval_ar_question_splits.sh
```

Outputs:

```text
<run>/merged/ar_split_eval_parallel/
  split_00/
  ...
  split_04/
  merged/
    split_results.jsonl
    summary.json
```

## Notebook Mapping

Main single-model precomputed notebook:

- [ar_vs_spectrum_decay_precomputed.ipynb](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/notebooks/ar_vs_spectrum_decay_precomputed.ipynb)

Set:

```python
METRICS_DIR = Path('/workspace/outputs/offline_math500_temp1_k32/qwen_qwen3_4b_limit500/merged/analysis_parallel/merged')
```

Spectrum-only notebook:

- [trajectory_spectrum_decay_precomputed.ipynb](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/notebooks/trajectory_spectrum_decay_precomputed.ipynb)

Only use this if you explicitly ran the spectrum-only branch.

Split-eval notebook:

- [ar_question_split_eval.ipynb](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/notebooks/ar_question_split_eval.ipynb)

Set:

```python
RESULTS_DIR = Path('/workspace/outputs/offline_math500_temp1_k32/qwen_qwen3_4b_limit500/merged/ar_split_eval_parallel/merged')
```

All-model combined notebook:

- [ar_vs_spectrum_decay_all_models_precomputed.ipynb](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/notebooks/ar_vs_spectrum_decay_all_models_precomputed.ipynb)

This notebook loads multiple per-model `analysis_parallel/merged/metrics.jsonl` files and combines them.

## Common Failure Modes

### 1. Host path vs container path confusion

Symptoms:

- code inside the container tries to read or create paths under `/network/...`

Meaning:

- host repo paths were not remapped to `/workspace/...`

Current fix:

- shell side uses [offline_common.sh](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/scripts/slurm/offline_common.sh)
- Python side uses [offline_utils.py](/network/scratch/p/pingsheng.li/Learning_from_Retrospection/research/scripts/offline_utils.py)

### 2. Corrupted HF cache during parallel startup

Symptoms:

- `JSONDecodeError`
- `InvalidHeaderDeserialization`

Meaning:

- multiple jobs tried to populate the same HF cache simultaneously

Current fix:

- end-to-end wrapper submits one predownload/export job first
- shard jobs start only after that job succeeds

### 3. AR training OOM

Meaning:

- too many train sequences in RAM

Current fix:

- use `SPILL_TRAIN_SEQUENCES=1`

### 4. Missing merged outputs

Check whether:

- collection shards exist,
- collection merge ran,
- AR metric shards exist,
- metric merge ran.

The canonical merged output should be:

```text
.../analysis_parallel/merged/metrics.jsonl
```

## Recommended Defaults

For the current main offline workflow:

- use the end-to-end wrapper,
- treat `analysis_parallel/merged/metrics.jsonl` as the canonical output,
- only run split-eval when you specifically need held-out AR results,
- only run spectrum-only when you specifically need that lighter branch.

That is the cleanest current usage pattern.
