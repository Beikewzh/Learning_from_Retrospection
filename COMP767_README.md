# COMP767 Final Experiment Guide

This file documents the final COMP767 experiment setup in this repo.

## Folder Layout

Tamia scripts are now split into:

- [scripts/tamia/final](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final)
  - active COMP767 final experiments
- [scripts/tamia/prm](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/prm)
  - PRM-related launchers kept for reference / optional reruns
- [scripts/tamia/legacy](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/legacy)
  - exploratory, older, or no-longer-used launchers

## Final Suite

The final Tamia comparison suite is launched with:

```bash
cd /scratch/p/psli/Learning_from_Retrospection
bash scripts/tamia/final/submit_full_suite_4gpu.sh
```

This submits 4 seeds (`1 2 3 4`) for each final method.

If you want custom seeds:

```bash
bash scripts/tamia/final/submit_full_suite_4gpu.sh 5 6 7 8
```

## Shared Training Setup

All final Tamia runs are aligned to:

- `4` GPUs
- `24` CPUs
- `240G` host RAM
- `worker.rollout.n=8`
- `data.rollout_batch_size=16`
- `worker.actor.global_batch_size=16`
- `data.max_response_length=1024`
- `trainer.max_steps=150`
- validation every `5` steps
- periodic checkpoint saving disabled with `trainer.save_freq=-1`

Most jobs request `24:00:00`. The two tanh reruns that had scheduling issues currently request `12:00:00`:

- [math_length_lora_4gpu_group_zscore_tanh.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_length_lora_4gpu_group_zscore_tanh.sbatch)
- [math_entropy_lora_4gpu_group_zscore_tanh.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_entropy_lora_4gpu_group_zscore_tanh.sbatch)

## Final Methods

The final suite contains:

1. Vanilla GRPO
2. Correct-only group-length baseline
3. `L1-Exact`
4. `L1-Max`
5. LeaRS + group z-score
6. LeaRS + group z-score + tanh
7. Length intrinsic + group z-score
8. Length intrinsic + group z-score + tanh
9. Entropy intrinsic + group z-score
10. Entropy intrinsic + group z-score + tanh

### Launchers

- Vanilla:
  - [math_grpo_lora_4gpu.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_grpo_lora_4gpu.sbatch)
- Correct-only group-length:
  - [math_grpo_lora_4gpu_correct_length_group_penalty.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_grpo_lora_4gpu_correct_length_group_penalty.sbatch)
- `L1-Exact`:
  - [math_grpo_lora_4gpu_l1_exact.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_grpo_lora_4gpu_l1_exact.sbatch)
- `L1-Max`:
  - [math_grpo_lora_4gpu_l1_max.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_grpo_lora_4gpu_l1_max.sbatch)
- LeaRS group-zscore:
  - [math_lears_lora_4gpu_group_zscore.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_lears_lora_4gpu_group_zscore.sbatch)
- LeaRS group-zscore + tanh:
  - [math_lears_lora_4gpu_group_zscore_tanh.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_lears_lora_4gpu_group_zscore_tanh.sbatch)
- Length intrinsic group-zscore:
  - [math_length_lora_4gpu_group_zscore.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_length_lora_4gpu_group_zscore.sbatch)
- Length intrinsic group-zscore + tanh:
  - [math_length_lora_4gpu_group_zscore_tanh.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_length_lora_4gpu_group_zscore_tanh.sbatch)
- Entropy group-zscore:
  - [math_entropy_lora_4gpu_group_zscore.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_entropy_lora_4gpu_group_zscore.sbatch)
- Entropy group-zscore + tanh:
  - [math_entropy_lora_4gpu_group_zscore_tanh.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/math_entropy_lora_4gpu_group_zscore_tanh.sbatch)

## Final Hyperparameters

### Reward baselines

- Correct-only group-length baseline:
  - `alpha=0.01`
  - reward applies only to correct responses
  - length z-score computed within rollout group
- `L1-Exact`:
  - `alpha=0.01`
  - `target_length=512`
- `L1-Max`:
  - `alpha=0.01`
  - `target_length=512`
  - `delta=0.5`

### Intrinsic baselines

- LeaRS group-zscore:
  - intrinsic source: AR prediction error
  - normalization: pooled per-question group z-score
  - `eta=0.01`
- LeaRS group-zscore + tanh:
  - same as above
  - intrinsic transform: `tanh`
  - `eta=0.01`
- Length group-zscore:
  - intrinsic source: response length
  - normalization: pooled per-question group z-score
  - `eta=0.01`
- Length group-zscore + tanh:
  - same as above
  - intrinsic transform: `tanh`
  - `eta=0.01`
- Entropy group-zscore:
  - intrinsic source: sampled-token surprise / entropy proxy
  - normalization: pooled per-question group z-score
  - `eta=0.01`
- Entropy group-zscore + tanh:
  - same as above
  - intrinsic transform: `tanh`
  - `eta=0.01`

### LeaRS AR schedule

The final LeaRS group-zscore runs use a lighter AR schedule:

- `research.ar.train_steps=30`
- `research.ar.train_every_n_steps=5`

## Per-Method Submitters

If you want to rerun only one family:

- Vanilla:
  - [submit_vanilla_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_vanilla_4gpu.sh)
- Correct-only group-length:
  - [submit_correct_length_group_penalty_alpha0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_correct_length_group_penalty_alpha0p01_4gpu.sh)
- `L1-Exact`:
  - [submit_l1_exact_alpha0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_l1_exact_alpha0p01_4gpu.sh)
- `L1-Max`:
  - [submit_l1_max_alpha0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_l1_max_alpha0p01_4gpu.sh)
- LeaRS group-zscore:
  - [submit_lears_eta0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_lears_eta0p01_4gpu.sh)
- LeaRS group-zscore + tanh:
  - [submit_lears_groupz_tanh_eta0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_lears_groupz_tanh_eta0p01_4gpu.sh)
- Length group-zscore:
  - [submit_length_groupz_eta0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_length_groupz_eta0p01_4gpu.sh)
- Length group-zscore + tanh:
  - [submit_length_groupz_tanh_eta0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_length_groupz_tanh_eta0p01_4gpu.sh)
- Entropy group-zscore:
  - [submit_entropy_group_zscore_eta0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_entropy_group_zscore_eta0p01_4gpu.sh)
- Entropy group-zscore + tanh:
  - [submit_entropy_groupz_tanh_eta0p01_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_entropy_groupz_tanh_eta0p01_4gpu.sh)

## Monitoring

Queue:

```bash
squeue -u $USER -o "%.18i %.12P %.30j %.8T %.10M %.20l %.30R"
```

History:

```bash
sacct -u $USER --format=JobID,JobName%30,State,Elapsed,ReqMem,MaxRSS,AllocTRES,ExitCode
```

## Metrics Export

### Export full run histories from `experiment_log.jsonl`

Use:

```bash
python scripts/export_experiment_metrics.py --output-prefix /tmp/final_suite_metrics
```

This writes:

- `/tmp/final_suite_metrics.json`
- `/tmp/final_suite_metrics.csv`

### Summarize W&B-exported CSVs

Accuracy CSV:

```bash
python scripts/summarize_accuracy_csv.py accuracy.csv --steps 100 120 125 150
```

Response length CSV:

```bash
python scripts/summarize_accuracy_csv.py response_length.csv --steps 100 120 125 150
```

The script prints lines in the format:

```text
method (n=4): mean +- std, min=..., max=...
```

## Game Of 24

There is also a separate vanilla Game of 24 launcher that reuses this repo's trainer with the vendored `vendor/24game` dataset:

```bash
cd /scratch/p/psli/Learning_from_Retrospection
bash scripts/tamia/final/submit_game24_vanilla_4gpu.sh
```

There is now also a full Game of 24 suite matching the math-suite training budget:

```bash
cd /scratch/p/psli/Learning_from_Retrospection
bash scripts/tamia/final/submit_game24_full_suite_4gpu.sh
```

Files:

- launcher:
  - [game24_grpo_lora_4gpu.sbatch](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/game24_grpo_lora_4gpu.sbatch)
- submitter:
  - [submit_game24_vanilla_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_game24_vanilla_4gpu.sh)
- prompt template:
  - [game24.jinja](/scratch/p/psli/Learning_from_Retrospection/examples/format_prompt/game24.jinja)
- reward:
  - [game24.py](/scratch/p/psli/Learning_from_Retrospection/examples/reward_function/game24.py)
- full-suite entrypoint:
  - [submit_game24_full_suite_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_game24_full_suite_4gpu.sh)

Integration choice:

- this repo's `verl` stays the trainer
- vendored `vendor/24game/verl` is not used
- data is read directly from `vendor/24game/data/24game_grpo/{train,val}.parquet`
- `question` is used as the prompt text
- `reward_model` is used as structured ground truth for correctness checking
- reward-side length baselines use `target_length=256` rather than `512`, because Game of 24 responses are much shorter than the math setting

## What The Rest Of The Scripts Are Doing

Everything outside [scripts/tamia/final](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final) is not part of the active COMP767 suite.

That includes:

- PRM launchers kept separately in [scripts/tamia/prm](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/prm)
- older tutorial launchers in [scripts/tamia/legacy](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/legacy)
- older length-penalty baselines
- running-zscore intrinsic variants
- sigmoid-only ablations that are not in the final suite
- batch-zscore exploratory variants
- Fir/Rorqual port scripts
- one-off helper scripts created during tuning/debugging

The practical rule is:

- if a script is called by [submit_full_suite_4gpu.sh](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/final/submit_full_suite_4gpu.sh), it is part of the final COMP767 suite
- if it lives under [scripts/tamia/prm](/scratch/p/psli/Learning_from_Retrospection/scripts/tamia/prm), it is PRM-specific but not in the final suite
- otherwise, treat it as exploratory or legacy unless you specifically need it
