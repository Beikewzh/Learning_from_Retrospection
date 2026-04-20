import math
import re
from collections import defaultdict
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "math_correct_length_group_penalty"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    if var <= 0:
        return [0.0 for _ in values]
    std = var ** 0.5
    return [(v - mean) / std for v in values]


def compute_score(
    reward_inputs: list[dict[str, Any]],
    alpha: float = 0.05,
    group_size: int = 5,
) -> list[dict[str, float]]:
    scores: list[dict[str, float] | None] = [None] * len(reward_inputs)
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, reward_input in enumerate(reward_inputs):
        group_key = (str(reward_input["source_prompt"]), str(reward_input["ground_truth"]))
        grouped_indices[group_key].append(idx)

    for group in grouped_indices.values():
        if len(group) > group_size:
            raise ValueError(f"Found prompt group larger than configured group_size={group_size}")

        group_inputs = [reward_inputs[idx] for idx in group]
        group_correct_lengths = []
        group_correct_positions = []
        for pos, reward_input in enumerate(group_inputs):
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
            if accuracy_score > 0:
                group_correct_lengths.append(float(reward_input["response_length"]))
                group_correct_positions.append(pos)

        correct_zscores = _zscore(group_correct_lengths)
        z_by_position = {pos: z for pos, z in zip(group_correct_positions, correct_zscores)}

        for pos, (idx, reward_input) in enumerate(zip(group, group_inputs)):
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
            format_score = format_reward(response)
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
            z_length = z_by_position.get(pos, 0.0)
            sigmoid_z = 1.0 / (1.0 + math.exp(-z_length)) if accuracy_score > 0 else 0.0
            correct_length_reward = 1.0 - alpha * sigmoid_z if accuracy_score > 0 else 0.0
            overall = correct_length_reward if accuracy_score > 0 else 0.0

            scores[idx] = {
                "overall": overall,
                "accuracy": accuracy_score,
                "format": format_score,
                "z_length": z_length,
                "sigmoid_z_length": sigmoid_z,
                "correct_length_reward": correct_length_reward,
            }

    return [score for score in scores if score is not None]
