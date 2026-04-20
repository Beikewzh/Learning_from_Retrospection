# Dataset Statistics

This file summarizes the dataset-side statistics used in the project for the Math and Game of 24 tasks.

## Math

### Train Set: `math12k`

Source:
- [train.parquet](/scratch/p/psli/Learning_from_Retrospection/data/math12k/train.parquet)

Available columns:
- `problem`
- `answer`

Summary:

| Split | Examples | Avg. input length (chars) | Max input length (chars) | Avg. answer length (chars) | Max answer length (chars) | Metadata available |
|---|---:|---:|---:|---:|---:|---|
| `math12k` train | 12000 | 202.90 | 4309 | 6.31 | 159 | No subject/level fields |

Note:
- the Math train set does not contain explicit difficulty or subject metadata

### Eval Set: `math500`

Source:
- [test.parquet](/scratch/p/psli/Learning_from_Retrospection/data/math500/test.parquet)

Available columns:
- `problem`
- `solution`
- `answer`
- `subject`
- `level`
- `unique_id`

Summary:

| Split | Examples | Avg. input length (chars) | Max input length (chars) | Avg. answer length (chars) | Max answer length (chars) |
|---|---:|---:|---:|---:|---:|
| `math500` test | 500 | 195.89 | 1733 | 5.93 | 53 |

#### Math500 Level Breakdown

| Level | Count |
|---|---:|
| 1 | 43 |
| 2 | 90 |
| 3 | 105 |
| 4 | 128 |
| 5 | 134 |

#### Math500 Subject Breakdown

| Subject | Count |
|---|---:|
| Algebra | 124 |
| Intermediate Algebra | 97 |
| Prealgebra | 82 |
| Number Theory | 62 |
| Precalculus | 56 |
| Geometry | 41 |
| Counting & Probability | 38 |

## Game of 24

Sources:
- [train.parquet](/scratch/p/psli/Learning_from_Retrospection/vendor/24game/data/24game_grpo/train.parquet)
- [val.parquet](/scratch/p/psli/Learning_from_Retrospection/vendor/24game/data/24game_grpo/val.parquet)

Available columns:
- `prompt_content`
- `reasoning_content`
- `answer`
- `result`
- `question`
- `is_possible`
- `data_source`
- `prompt`
- `ability`
- `reward_model`
- `extra_info`

### Split Summary

| Split | Examples | Avg. input length (chars) | Max input length (chars) | Avg. answer length (chars) | Max answer length (chars) |
|---|---:|---:|---:|---:|---:|
| train | 1638 | 1170.23 | 1173 | 15.52 | 46 |
| val/test | 182 | 1170.20 | 1173 | 15.98 | 41 |

### Solvable vs. Unsolvable Breakdown

| Split | Total | Solvable | Unsolvable | Solvable ratio | Unsolvable ratio |
|---|---:|---:|---:|---:|---:|
| train | 1638 | 1216 | 422 | 0.7424 | 0.2576 |
| val/test | 182 | 146 | 36 | 0.8022 | 0.1978 |

## Interpretation

Useful takeaways for the report:

- Math training uses a much larger dataset (`12k`) than Game of 24 (`1638`).
- Math prompts are shorter on average than Game of 24 prompts.
- In both tasks, reference answers are very short relative to generated reasoning traces.
- Game of 24 is imbalanced toward solvable examples, especially in validation.
- Math500 evaluation includes explicit difficulty (`level`) and subject annotations, while the Math train set does not.
