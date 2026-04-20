import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "math_l1_max_length"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    alpha: float = 0.01,
    target_length: int = 1024,
    delta: float = 0.5,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        response_length = int(reward_input["response_length"])
        unclipped = alpha * (float(target_length) - float(response_length)) + float(delta)
        max_factor = max(0.0, min(1.0, unclipped))
        overall = accuracy_score * max_factor

        scores.append(
            {
                "overall": overall,
                "accuracy": accuracy_score,
                "format": format_score,
                "target_length": float(target_length),
                "response_length_value": float(response_length),
                "max_factor": max_factor,
                "max_unclipped": unclipped,
            }
        )

    return scores
