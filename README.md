# Learning from Retrospection

Research code for studying whether retrospective signals from reasoning trajectories can improve reinforcement learning for language-model reasoning.

This repository builds on EasyR1/veRL and adds the project-specific machinery for **LeaRS**: latent trajectory collection, lightweight autoregressive latent modeling, token-level intrinsic reward shaping, and offline trajectory-complexity analysis.

## Project Overview

The project has two connected tracks:

1. **Online RL for reasoning**
   - Train a Qwen3-4B policy with GRPO on Math and Game of 24.
   - Compare vanilla GRPO, scalar length-reward baselines, and token-level intrinsic baselines.
   - LeaRS uses online latent autoregressive prediction error as an intrinsic signal inside the GRPO advantage.

2. **Offline latent analysis**
   - Sample multiple reasoning trajectories from frozen models.
   - Extract response-token hidden-state trajectories.
   - Train small AR predictors over latents.
   - Analyze response length, eigenspectrum decay, and AR prediction error against rollout success.

The detailed method writeups are:

- [METHOD_ONLINE_RL.md](METHOD_ONLINE_RL.md)
- [METHOD_OFFLINE_ANALYSIS.md](METHOD_OFFLINE_ANALYSIS.md)
- [DATASET_STATS.md](DATASET_STATS.md)

## Repository Layout

| Path | Purpose |
|---|---|
| `verl/` | EasyR1/veRL training stack with project-specific hooks |
| `research/` | LeaRS modules, latent buffers, AR models, intrinsic reward logic, offline scripts |
| `examples/research/` | Hydra configs for Math LeaRS/GRPO experiments |
| `examples/reward_function/` | Math and Game of 24 reward functions and baselines |
| `scripts/tamia/final/` | Final Tamia/Slurm experiment launchers |
| `scripts/slurm/` | General Slurm entrypoints for LeaRS and offline analysis |
| `research/offline/` | Offline latent collection and analysis workflows |
| `research/notebooks/` | Analysis notebooks for trajectory metrics and experiment results |
| `data/`, `vendor/24game/` | Local datasets and Game of 24 assets |
| `checkpoints/`, `outputs/`, `wandb/` | Generated experiment artifacts |

## Environment

The intended runtime is the EasyR1 container:

```bash
hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

On HPC systems, pull it as a Singularity/Apptainer image:

```bash
cd /path/to/Learning_from_Retrospection
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

Then enter the container with the repository bound to `/workspace`:

```bash
apptainer shell --nv --cleanenv \
  --bind "${PWD}:/workspace" \
  easyr1.sif

cd /workspace
pip install -e .
```

For site-specific notes, see [INSTALL_HPC.md](INSTALL_HPC.md).

## Data Setup

The main Math experiments use:

- `hiyouga/math12k` for training,
- `HuggingFaceH4/MATH-500` for evaluation.

Download the default model and Math datasets with the commands in [download_dataset.md](download_dataset.md). The expected local paths are:

```text
models/Qwen__Qwen3-4B/
data/math12k/train.parquet
data/math500/test.parquet
```

Game of 24 data is stored under:

```text
vendor/24game/data/24game_grpo/
```

## Quick Smoke Test

Inside the container at `/workspace`, run a 1-GPU LeaRS sanity check:

```bash
ENABLE_WANDB=1 \
TOTAL_STEPS=4 WARMUP_STEPS=1 AR_TRAIN_EVERY_N_STEPS=1 \
MAX_PROMPT_LENGTH=128 MAX_RESPONSE_LENGTH=128 \
ROLLOUT_BATCH_SIZE=1 ACTOR_GLOBAL_BATCH_SIZE=1 ROLLOUT_N=2 \
AR_TRAIN_STEPS=1 AR_MIN_BUFFER_SAMPLES=1 AR_BATCH_SIZE=1 \
bash scripts/lears_smoke_1gpu.sh
```

For the complete online LeaRS interface, see:

- [docs/research/lears_quickstart.md](docs/research/lears_quickstart.md)
- [docs/research/lears.md](docs/research/lears.md)

## Online RL Experiments

Final Math comparison suite:

```bash
cd /path/to/Learning_from_Retrospection
bash scripts/tamia/final/submit_full_suite_4gpu.sh
```

By default this submits seeds `1 2 3 4`. To choose seeds manually:

```bash
bash scripts/tamia/final/submit_full_suite_4gpu.sh 5 6 7 8
```

The final suite covers:

- vanilla GRPO,
- correct-only group-length reward shaping,
- L1-Exact,
- L1-Max,
- LeaRS intrinsic shaping,
- response-length intrinsic shaping,
- entropy/surprise intrinsic shaping,
- identity and tanh variants for the token-level intrinsic methods.

Game of 24 launchers live in the same directory:

```bash
scripts/tamia/final/game24_*.sbatch
scripts/tamia/final/submit_game24_full_suite_4gpu.sh
```

The course-facing experiment guide is [COMP767_README.md](COMP767_README.md).

## Offline Analysis

The canonical offline path collects MATH-500 latent trajectories, trains an AR model over the merged latent buffer, computes AR+spectrum metrics, and writes:

```text
<run>/merged/analysis_parallel/merged/metrics.jsonl
```

Example one-model run:

```bash
MODEL_ID=Qwen/Qwen3-4B \
MODEL_TAG=qwen_qwen3_4b \
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

For the full workflow, see:

- [research/offline/OFFLINE_PIPELINE.md](research/offline/OFFLINE_PIPELINE.md)
- [research/offline/README.md](research/offline/README.md)
- [research/offline/game24/README.md](research/offline/game24/README.md)

## Metrics and Results

Online runs write trainer logs, checkpoints, and research traces under the configured checkpoint/output roots. W&B is enabled in the main experiment configs.

Export JSONL experiment logs to JSON/CSV:

```bash
python scripts/export_experiment_metrics.py --output-prefix /tmp/final_suite_metrics
```

Summarize exported W&B CSVs:

```bash
python scripts/summarize_accuracy_csv.py accuracy.csv --steps 100 120 125 150
```

## Tests

Run focused research tests with:

```bash
pytest tests/research
```

For a smaller check while iterating on offline pipelines:

```bash
pytest tests/research/test_ar_trainer.py tests/research/test_ar_scorer.py
```

## Acknowledgements

This repository is a research fork of EasyR1, which itself builds on veRL. The base training framework, distributed rollout path, and many examples come from those projects; the project-specific additions live mainly under `research/`, `examples/research/`, `examples/reward_function/`, and the experiment scripts.

See [LICENSE](LICENSE) for license details.
