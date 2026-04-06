import re
from typing import Any

from research.offline.game24.score_game24 import score_game24_response


REWARD_NAME = "game24"
REWARD_TYPE = "batch"

_CARD_LINE_RE = re.compile(r"cards\s*:\s*([^\n\r]+)", flags=re.IGNORECASE)
_DIGIT_RE = re.compile(r"-?\d+")


def _normalize_response_text(response: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", response).strip()


def _extract_cards_from_prompt(source_prompt: str, prompt: str) -> list[int]:
    text = source_prompt or prompt or ""
    m = _CARD_LINE_RE.search(text)
    if m:
        nums = [int(x) for x in _DIGIT_RE.findall(m.group(1))]
        if len(nums) >= 4:
            return nums[:4]

    nums = [int(x) for x in _DIGIT_RE.findall(text)]
    if len(nums) >= 4:
        return nums[-4:]
    raise ValueError("Could not parse 4 cards from prompt text")


def _is_possible_from_ground_truth(ground_truth: str) -> bool:
    gt = (ground_truth or "").strip().upper().strip(". ")
    return gt != "NO"


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        response = _normalize_response_text(str(reward_input.get("response", "")))
        ground_truth = str(reward_input.get("ground_truth", ""))

        try:
            cards = _extract_cards_from_prompt(
                source_prompt=str(reward_input.get("source_prompt", "")),
                prompt=str(reward_input.get("prompt", "")),
            )
            is_possible = _is_possible_from_ground_truth(ground_truth)
            result = score_game24_response(
                response=response,
                cards=cards,
                is_possible=is_possible,
                ground_truth=ground_truth,
            )
            accuracy = float(result["score_accuracy"])
            fmt = float(result["score_format"])
        except Exception:
            accuracy = 0.0
            fmt = 0.0

        overall = (1.0 - format_weight) * accuracy + format_weight * fmt
        scores.append(
            {
                "overall": overall,
                "accuracy": accuracy,
                "format": fmt,
            }
        )

    return scores
