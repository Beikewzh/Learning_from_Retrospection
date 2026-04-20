import re
from typing import Any

from examples.reward_function.game24 import accuracy_reward, format_reward


REWARD_NAME = "game24_l1_max_length"
REWARD_TYPE = "batch"


def compute_score(
    reward_inputs: list[dict[str, Any]],
    alpha: float = 0.01,
    target_length: int = 256,
    delta: float = 0.5,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"], reward_input["source_prompt"])
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
