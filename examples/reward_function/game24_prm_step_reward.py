import math
import re
from typing import Any

from research.offline.game24.score_game24 import score_game24_response


REWARD_NAME = "game24_prm_step_reward"
REWARD_TYPE = "batch"

_CARD_LINE_RE = re.compile(r"cards\s*:\s*([^\n\r]+)", flags=re.IGNORECASE)
_DIGIT_RE = re.compile(r"-?\d+")
_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
_STEP_SPLIT_RE = re.compile(r"\n\s*\n+")


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


def _extract_steps(response: str, min_step_chars: int) -> list[str]:
    m = _THINK_RE.search(response)
    if not m:
        return []
    content = m.group(1).strip()
    if not content:
        return []
    steps = [s.strip() for s in _STEP_SPLIT_RE.split(content) if s.strip()]
    return [s for s in steps if len(s) >= min_step_chars]


def _step_quality(step: str) -> float:
    has_number = bool(re.search(r"\d", step))
    has_op = bool(re.search(r"[+\-*/×÷]", step))
    has_eq = "=" in step
    score = 0.0
    if has_number:
        score += 0.4
    if has_op:
        score += 0.4
    if has_eq:
        score += 0.2
    return float(score)


def compute_score(
    reward_inputs: list[dict[str, Any]],
    prm_model_path: str = "",
    prm_weight: float = 0.1,
    format_weight: float = 0.1,
    max_prm_steps: int = 8,
    min_step_chars: int = 8,
) -> list[dict[str, float]]:
    # prm_model_path is accepted for compatibility with existing PRM scripts.
    _ = prm_model_path

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

        raw_score = (1.0 - format_weight) * accuracy + format_weight * fmt

        steps = _extract_steps(response, min_step_chars=min_step_chars)[: int(max_prm_steps)]
        step_scores = [_step_quality(s) for s in steps]

        prm_process_product = math.prod(step_scores) if step_scores else 0.0
        prm_process_min = min(step_scores) if step_scores else 0.0
        prm_process_last = step_scores[-1] if step_scores else 0.0
        prm_process_mean = (sum(step_scores) / len(step_scores)) if step_scores else 0.0

        scores.append(
            {
                "overall": raw_score + float(prm_weight) * prm_process_product,
                "accuracy": accuracy,
                "format": fmt,
                "raw_score": float(raw_score),
                "prm_process_product": float(prm_process_product),
                "prm_process_min": float(prm_process_min),
                "prm_process_last": float(prm_process_last),
                "prm_process_mean": float(prm_process_mean),
                "num_prm_steps": float(len(step_scores)),
            }
        )

    return scores
