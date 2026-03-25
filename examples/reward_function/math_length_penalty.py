import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math_length_penalty"
REWARD_TYPE = "batch"

# Reference token count used for scaling: a response of this length yields reward ≈ accuracy_score.
# Shorter → higher reward, longer → lower reward.  GRPO normalises within-group anyway, so the
# absolute scale does not matter; only the relative ordering across rollouts matters.
_TOKEN_REF = 256


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

        raw = (1 - format_weight) * accuracy_score + format_weight * format_score

        # T ≈ token count (chars / 4 is a standard rough approximation).
        # Divide raw score by (T / T_ref) so shorter correct responses score higher.
        # Wrong responses stay at 0 regardless of length.
        T = max(len(response) / 4.0, 1.0)
        overall = raw * _TOKEN_REF / T

        scores.append(
            {
                "overall": overall,          # length-penalised; used as GRPO reward
                "accuracy": accuracy_score,  # raw correctness, logged for analysis
                "format": format_score,
                "length_tokens_approx": T,   # approximate token count, for diagnostics
                "raw_score": raw,            # pre-penalty score, for diagnostics
            }
        )

    return scores
