# FIR Game24 Tutorial SBatch Scripts

This folder contains isolated Game24 variants of the Tamia tutorial scripts.

Files:
- `tutorial_game24_grpo_lora_4gpu.sbatch`
- `tutorial_game24_grpo_lora_4gpu_length_penalty.sbatch`
- `tutorial_game24_prm_4gpu.sbatch`
- `tutorial_game24_lears_lora_4gpu.sbatch`
- `tutorial_math_grpo_lora_4gpu.sbatch` (compat symlink)
- `tutorial_math_grpo_lora_4gpu_length_penalty.sbatch` (compat symlink)
- `tutorial_math_prm_4gpu.sbatch` (compat symlink)
- `tutorial_math_lears_lora_4gpu.sbatch` (compat symlink)

Defaults:
- `MODEL_PATH=/workspace/models/Qwen__Qwen3-4B`
- `GAME24_TRAIN_FILES=/workspace/data/game24_reasoning/game24_reasoning_limit500.jsonl`
- `GAME24_VAL_FILES=/workspace/data/game24_reasoning/game24_reasoning_limit500.jsonl`

Required supporting files added:
- `examples/format_prompt/game24.jinja`
- `examples/format_prompt/game24_prm_steps.jinja`
- `examples/reward_function/game24.py`
- `examples/reward_function/game24_length_penalty.py`
- `examples/reward_function/game24_prm_step_reward.py`

Example submit:
```bash
sbatch scripts/fir/game24/tutorial_game24_grpo_lora_4gpu.sbatch
```
