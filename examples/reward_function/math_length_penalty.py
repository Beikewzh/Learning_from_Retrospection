import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math_length_penalty"
REWARD_TYPE = "batch"

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    length_threshold: int = 512,
    length_penalty_per_token: float = 1e-4,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        raw_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        response_length = int(reward_input["response_length"])
        length_penalty = length_penalty_per_token * max(0, response_length - length_threshold)
        overall = raw_score - length_penalty

        scores.append(
            {
                "overall": overall,
                "accuracy": accuracy_score,
                "format": format_score,
                "length_penalty": length_penalty,
                "raw_score": raw_score,
            }
        )

    return scores
