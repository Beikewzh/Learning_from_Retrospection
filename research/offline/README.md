# Offline Latent Collection Workflow

This workflow is separate from the online LeaRS RL loop. It is for:

1. normalizing a reasoning dataset into a simple JSONL schema,
2. collecting latent trajectories from an existing pretrained or RLVR-posttrained model,
3. inspecting the resulting latent buffer before later offline analysis.

The online RL training path under `verl.trainer.main` is unchanged.

## Step 1. Prepare a normalized reasoning JSONL

Expected normalized schema:

```json
{"id": "amc_001", "question": "...", "answer": "..."}
```

If your source JSONL uses different keys, normalize it with:

```bash
python3 research/scripts/prepare_reasoning_jsonl.py \
  --input data/amc_raw.jsonl \
  --output data/amc_normalized.jsonl \
  --id-key problem_id \
  --question-key question \
  --answer-key answer
```

Notes:

- the script keeps extra fields under a `meta` field,
- blank lines are ignored,
- this step only rewrites JSONL, it does not tokenize or run any model.

## Step 2. Collect offline latent trajectories

This script:

1. loads a pretrained causal LM,
2. renders each question with the math prompt template,
3. generates one response per example,
4. scores it with a math-style correctness check,
5. runs a second forward pass with `output_hidden_states=True`,
6. extracts response-token latents from the chosen hidden layer,
7. stores latent trajectories in the existing Parquet latent buffer,
8. writes a sidecar `metadata.jsonl` with prompt/response/score fields,
9. records token-span boundaries for the `<think> ... </think>` region and boxed answer when present.

Example:

```bash
python3 research/scripts/collect_latents_offline.py \
  --model allenai/Llama-3.1-Tulu-3-8B \
  --input data/amc_normalized.jsonl \
  --output-dir outputs/tulu3_amc_latents \
  --cache-root .cache \
  --prompt-template examples/format_prompt/math.jinja \
  --layer-index -1 \
  --max-prompt-length 512 \
  --max-response-length 512 \
  --temperature 0.0 \
  --device cuda
```

Outputs:

- buffer shards: `outputs/tulu3_amc_latents/buffer/shards/*.parquet`
- buffer manifest: `outputs/tulu3_amc_latents/buffer/manifest.json`
- sidecar metadata: `outputs/tulu3_amc_latents/metadata.jsonl`
- collection summary: `outputs/tulu3_amc_latents/collection_summary.json`

Important implementation details:

- `--temperature 0.0` means greedy decoding,
- `--layer-index -1` means the last hidden state,
- only response-token latents are stored,
- `--cache-root .cache` keeps HF and dataset caches under the repo's scratch-backed `.cache/`,
- `metadata.jsonl` contains `think_token_start`, `think_token_end`, `answer_token_start`, and `answer_token_end`,
- `score_overall` currently equals exact-answer correctness,
- format correctness is also recorded for convenience.

## Step 3. Inspect the saved buffer

Use the existing buffer inspection script:

```bash
python3 research/scripts/inspect_buffer.py \
  --buffer-dir outputs/tulu3_amc_latents/buffer \
  --show 5
```

This prints:

- total number of saved samples,
- number of Parquet shards,
- a few sample rows with lengths, hidden dimension, score label, and blob size.

## What this workflow does not do

- it does not run RL,
- it does not train the AR model in-loop,
- it does not compute intrinsic reward,
- it does not change any existing online LeaRS training files.

## Suggested next step after collection

Once collection looks correct, you can train the existing offline AR model with:

```bash
python3 research/scripts/train_ar_offline.py \
  --buffer-dir outputs/tulu3_amc_latents/buffer \
  --output-dir outputs/tulu3_amc_latents/ar \
  --device cpu
```
