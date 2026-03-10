# LeaRS (Research Feature Flag)

LeaRS is an optional research workflow for latent-space self-supervised signals during PPO training.

Default behavior is unchanged:

- `research.enabled=false`: baseline EasyR1 behavior.
- `research.enabled=true`: latent capture + buffer + periodic AR training + intrinsic reward fusion.

## What changes when enabled

1. `vLLM` still handles rollout generation.
2. Actor log-prob recomputation additionally captures last-layer hidden states on response tokens.
3. Latents are written to a Parquet buffer under the run checkpoint directory (or `research.buffer.dir`).
4. A tiny AR model is trained offline every `research.ar.train_every_n_steps`.
5. AR prediction error is transformed into intrinsic reward and added into `token_level_scores` before advantage computation.

## Main config tree

Use the `research.*` config namespace in YAML/CLI:

- `research.enabled`
- `research.latent.*`
- `research.buffer.*`
- `research.ar.*`
- `research.intrinsic.*`

Current limitation:

- `research.latent.include_prompt` must be `false`.

Reference defaults are visible in:

- `examples/config.yaml` (commented block)
- `examples/research/lears_gsm8k.yaml` (end-to-end sample)

## Dataset string format

EasyR1 dataset fields (`data.train_files`, `data.val_files`) support:

- `dataset@split` (existing behavior)
- `dataset:config@split` (new, for multi-config HF datasets)
- `dataset#config@split` (equivalent alias)

Example for GSM8K:

- `openai/gsm8k:main@train`
- `openai/gsm8k:main@test`

## Run example

```bash
python3 -m verl.trainer.main config=examples/research/lears_gsm8k.yaml
```

To keep baseline behavior with the same config file:

```bash
python3 -m verl.trainer.main config=examples/research/lears_gsm8k.yaml research.enabled=false
```

## Standalone AR utilities

Inspect buffer:

```bash
python3 research/scripts/inspect_buffer.py --buffer-dir /path/to/checkpoints/.../research/buffer
```

Train AR offline from a saved buffer:

```bash
python3 research/scripts/train_ar_offline.py \
  --buffer-dir /path/to/checkpoints/.../research/buffer \
  --output-dir /path/to/checkpoints/.../research/ar \
  --device cpu
```

## Logging

When enabled, additional metrics are emitted with `research/` prefixes, including:

- latent capture counters
- buffer shard/sample stats
- AR training status/loss
- intrinsic signal stats and activation state

## Troubleshooting

- If AR is not yet trained or there are not enough buffer samples, intrinsic defaults to zero.
- `research.latent.include_prompt=true` is currently unsupported; set it to `false`.
- For GPUs without BF16 support, set rollout/FSDP precision to FP16 in your config.
- For pre-Ampere GPUs (e.g., RTX 8000, compute capability 7.5), disable padding-free mode (`worker.actor.padding_free=false`).
- If disk usage grows too fast, lower `research.buffer.max_disk_gb` or `research.buffer.shard_max_samples`.
