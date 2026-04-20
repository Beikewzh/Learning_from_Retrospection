import math
import re
from collections import defaultdict
from typing import Any

from examples.reward_function.game24 import accuracy_reward, format_reward


REWARD_NAME = "game24_correct_length_group_penalty"
REWARD_TYPE = "batch"


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    if var <= 0:
        return [0.0 for _ in values]
    std = var**0.5
    return [(v - mean) / std for v in values]


def compute_score(
    reward_inputs: list[dict[str, Any]],
    alpha: float = 0.01,
    group_size: int = 8,
) -> list[dict[str, float]]:
    scores: list[dict[str, float] | None] = [None] * len(reward_inputs)
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, reward_input in enumerate(reward_inputs):
        grouped_indices[str(reward_input["source_prompt"])].append(idx)

    for group in grouped_indices.values():
        if len(group) > group_size:
            raise ValueError(f"Found prompt group larger than configured group_size={group_size}")

        group_inputs = [reward_inputs[idx] for idx in group]
        group_correct_lengths: list[float] = []
        group_correct_positions: list[int] = []

        for pos, reward_input in enumerate(group_inputs):
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"], reward_input["source_prompt"])
            if accuracy_score > 0:
                group_correct_lengths.append(float(reward_input["response_length"]))
                group_correct_positions.append(pos)

        correct_zscores = _zscore(group_correct_lengths)
        z_by_position = {pos: z for pos, z in zip(group_correct_positions, correct_zscores)}

        for pos, (idx, reward_input) in enumerate(zip(group, group_inputs)):
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
            format_score = format_reward(response)
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"], reward_input["source_prompt"])
            z_length = z_by_position.get(pos, 0.0)
            sigmoid_z = 1.0 / (1.0 + math.exp(-z_length)) if accuracy_score > 0 else 0.0
            correct_length_reward = 1.0 - alpha * sigmoid_z if accuracy_score > 0 else 0.0

            scores[idx] = {
                "overall": correct_length_reward if accuracy_score > 0 else 0.0,
                "accuracy": accuracy_score,
                "format": format_score,
                "z_length": z_length,
                "sigmoid_z_length": sigmoid_z,
                "correct_length_reward": correct_length_reward,
            }

    return [score for score in scores if score is not None]
