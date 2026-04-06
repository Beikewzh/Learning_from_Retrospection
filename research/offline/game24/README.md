# Game24 Offline Pipeline (Isolated)

This folder contains a Game24-specific offline pipeline that is isolated from the existing Math500 scripts.

## Components

- `prepare_game24_jsonl.py`
  - Input: upstream Game24 JSON/JSONL (for example `all_24_game_results_shuffled.json`)
  - Output: normalized JSONL with required keys:
    - `id`, `question`, `answer`, `cards`, `is_possible`
    - optional passthrough `meta`
- `score_game24.py`
  - Strict equation checker for solvable cases.
  - `is_possible=false` requires normalized `NO`.
- `collect_latents_game24_offline.py`
  - Collects response-token latents + metadata.
  - Emits downstream-required metadata keys:
    - `uid`, `question_uid`, `sample_idx`, `response_length`, `success`, `score_accuracy`, `think_token_start`, `think_token_end`
- `merge_game24_shards.py`
  - Merges shard outputs into `merged/` with `uid` dedup.
  - Supports both `shard_XX` layout and single-shard layout.
- `train_game24_ar_on_run.py`
- `compute_game24_ar_spectrum_metrics.py`
- `merge_game24_analysis_metrics.py`
  - Produces canonical `analysis_parallel/merged/metrics.jsonl`.

## SLURM Entry Point

Use:

```bash
scripts/slurm/game24/submit_game24_model_end_to_end.sh
```

Required env (minimum):

- `MODEL_ID`

Commonly configured env:

- Core: `MODEL_TAG`, `CACHE_ROOT`, `OUTPUT_BASE`, `LIMIT`
- Sampling: `NUM_SHARDS`, `NUM_SAMPLES_PER_QUESTION`, `TEMPERATURE`, `TOP_P`, `TOP_K`
- Generation: `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `LAYER_INDEX`, `TORCH_DTYPE`, `LATENT_DTYPE`
- Data: `DATA_JSON_SOURCE`, `DATA_PATH`, `PROMPT_TEMPLATE`
- AR/metrics: `USE_REASONING_SPAN`, `AR_DEVICE`, `AR_MAX_SAMPLES`, `AR_TRAIN_STEPS`, `AR_BATCH_SIZE`, `MAX_COMPONENTS`, `MIN_SEQ_LEN`
- Reliability: `RESUME`, `OVERWRITE`, `FLUSH_EVERY_N_SAMPLES`

## Stage Outputs

For one run root:

- Collection shards: `.../<model_tag>_limit<limit>/shard_XX/`
- Collection merge: `.../<model_tag>_limit<limit>/merged/`
- AR model: `.../merged/analysis_parallel/ar_model/`
- Metric shards: `.../merged/analysis_parallel/shards/metrics_shard_XX.jsonl`
- Canonical metrics:
  - `.../merged/analysis_parallel/merged/metrics.jsonl`

## Notebook

Game24 analysis notebook:

- `research/notebooks/game24/ar_spectrum_length_analysis_game24.ipynb`

It reads canonical merged metrics and joins `metadata.jsonl` for `is_possible` stratification if needed.
