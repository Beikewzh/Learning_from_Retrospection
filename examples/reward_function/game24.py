# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import Any


REWARD_NAME = "game24"
REWARD_TYPE = "batch"

FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
CARDS_PATTERN = re.compile(r"Cards:\s*([0-9,\s]+)\.?", re.IGNORECASE)
ALLOWED_EXPRESSION_PATTERN = re.compile(r"^[\d\s+\-*/().]+$")


def format_reward(response: str) -> float:
    return 1.0 if re.fullmatch(FORMAT_PATTERN, response.strip()) else 0.0


def extract_answer(response: str) -> str:
    match = ANSWER_PATTERN.search(response)
    candidate = match.group(1).strip() if match else response.strip()

    no_match = re.search(r"\bNO\b\.?", candidate, re.IGNORECASE)
    if no_match:
        return "NO"

    equation_match = re.search(r"([^<>\n]+?=\s*24)\s*$", candidate)
    if equation_match:
        return equation_match.group(1).strip()

    for line in reversed(candidate.splitlines()):
        line = line.strip()
        if not line:
            continue
        if re.search(r"\bNO\b\.?", line, re.IGNORECASE):
            return "NO"
        if "=" in line:
            return line
        return line

    return candidate


def parse_cards(source_prompt: str) -> list[int]:
    match = CARDS_PATTERN.search(source_prompt)
    if not match:
        return []

    return [int(value.strip()) for value in match.group(1).split(",") if value.strip()]


def parse_ground_truth(ground_truth: Any) -> tuple[bool, str]:
    metadata = ground_truth
    if isinstance(ground_truth, str):
        try:
            metadata = json.loads(ground_truth)
        except json.JSONDecodeError:
            metadata = {"ground_truth": ground_truth, "is_possible": ground_truth.strip().upper() != "NO"}

    if not isinstance(metadata, dict):
        raise TypeError(f"Unsupported Game24 ground truth type: {type(ground_truth)}")

    return bool(metadata.get("is_possible", False)), str(metadata.get("ground_truth", "")).strip()


def extract_numbers_from_expression(expression: str) -> list[int]:
    return [int(number) for number in re.findall(r"\d+", expression)]


def compute_equation(answer: str, cards: list[int]) -> bool:
    expression = answer.split("=", 1)[0].strip()
    expression = expression.replace("×", "*").replace("÷", "/")

    if "**" in expression or "//" in expression:
        return False

    if not re.fullmatch(ALLOWED_EXPRESSION_PATTERN, expression):
        return False

    if sorted(extract_numbers_from_expression(expression)) != sorted(cards):
        return False

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return False

    return abs(float(result) - 24.0) < 1e-6


def accuracy_reward(response: str, ground_truth: Any, source_prompt: str) -> float:
    is_possible, _ = parse_ground_truth(ground_truth)
    answer = extract_answer(response)

    if not is_possible:
        return 1.0 if answer == "NO" else 0.0

    cards = parse_cards(source_prompt)
    if answer == "NO" or not cards:
        return 0.0

    return 1.0 if compute_equation(answer, cards) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"], reward_input["source_prompt"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
