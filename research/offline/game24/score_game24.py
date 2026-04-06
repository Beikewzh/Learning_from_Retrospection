#!/usr/bin/env python3
"""Scoring helpers for Game24 responses."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{(.*)\}", flags=re.DOTALL)
FORMAT_PATTERN = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>.*", flags=re.DOTALL | re.IGNORECASE)


class EquationValidationError(ValueError):
    """Raised when an equation candidate is malformed."""


def normalize_response_text(text: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", text)


def parse_cards(cards_raw: Any) -> list[int]:
    if isinstance(cards_raw, (list, tuple)):
        cards = [int(x) for x in cards_raw]
    elif isinstance(cards_raw, str):
        cards = [int(x) for x in re.findall(r"-?\d+", cards_raw)]
    else:
        raise TypeError(f"Unsupported cards type: {type(cards_raw).__name__}")
    if len(cards) != 4:
        raise ValueError(f"Expected exactly 4 cards, got {cards}")
    return cards


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y", "possible", "solvable"}:
            return True
        if v in {"false", "f", "0", "no", "n", "impossible", "unsolvable"}:
            return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def extract_answer_candidate(response: str) -> str:
    normalized = normalize_response_text(response)
    match = ANSWER_TAG_PATTERN.search(normalized)
    if match is not None:
        return match.group(1).strip()

    boxed = BOXED_PATTERN.search(normalized)
    if boxed is not None:
        return boxed.group(1).strip()

    return normalized.strip()


def normalize_no_token(text: str) -> str:
    cleaned = re.sub(r"[`$]", "", text).strip()
    cleaned = cleaned.strip(". ")
    return cleaned.upper()


def normalize_equation_candidate(text: str) -> str:
    candidate = text.strip()
    candidate = candidate.replace("×", "*").replace("÷", "/")
    candidate = candidate.replace("−", "-")
    candidate = re.sub(r"(?i)^final\s+answer\s*[:：]\s*", "", candidate)
    candidate = re.sub(r"(?i)^answer\s*[:：]\s*", "", candidate)

    if "=" in candidate:
        candidate = candidate.split("=", 1)[0].strip()

    if re.search(r"[A-Za-z]", candidate):
        matches = re.findall(r"[\d\(\)\+\-\*\/\.\s]+", candidate)
        matches = [m.strip() for m in matches if m.strip()]
        matches = [m for m in matches if re.search(r"\d", m) and re.search(r"[\+\-\*/]", m)]
        if matches:
            candidate = max(matches, key=len)

    candidate = candidate.strip().strip(".;")
    return candidate


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise EquationValidationError("Division by zero")
            return left / right
        raise EquationValidationError(f"Unsupported operator: {type(op).__name__}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise EquationValidationError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise EquationValidationError("Only numeric constants are allowed")
        return float(node.value)
    raise EquationValidationError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_expression(expr: str) -> tuple[float, list[int]]:
    if not expr:
        raise EquationValidationError("Empty expression")
    if "**" in expr or "//" in expr or "%" in expr:
        raise EquationValidationError("Disallowed operator")
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        raise EquationValidationError("Expression contains invalid characters")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise EquationValidationError(f"Syntax error: {exc}") from exc

    numbers: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = float(node.value)
            if not math.isfinite(value):
                raise EquationValidationError("Non-finite numeric literal")
            if not value.is_integer():
                raise EquationValidationError("Numeric literals must be integers")
            numbers.append(int(value))

    value = _eval_ast(tree)
    if not math.isfinite(value):
        raise EquationValidationError("Non-finite expression value")
    return value, numbers


def score_game24_response(
    *,
    response: str,
    cards: Any,
    is_possible: Any,
    ground_truth: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_response_text(response)
    parsed_answer = extract_answer_candidate(response)
    cards_list = parse_cards(cards)
    possible_case_expected = parse_boolish(is_possible)

    format_ok = bool(FORMAT_PATTERN.fullmatch(normalized))
    no_token = normalize_no_token(parsed_answer) == "NO"

    equation_expr = None
    equation_value = None
    used_numbers: list[int] | None = None
    numbers_match = None
    equation_valid = False
    parse_error = None

    if not no_token:
        equation_expr = normalize_equation_candidate(parsed_answer)
        try:
            equation_value, used_numbers = evaluate_expression(equation_expr)
            numbers_match = sorted(used_numbers) == sorted(cards_list)
            equation_valid = numbers_match and abs(equation_value - 24.0) < 1e-6
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)

    if possible_case_expected:
        score_accuracy = 1.0 if equation_valid else 0.0
    else:
        score_accuracy = 1.0 if no_token else 0.0

    score_overall = score_accuracy
    success = bool(score_accuracy > 0.5)

    return {
        "score_accuracy": float(score_accuracy),
        "score_format": 1.0 if format_ok else 0.0,
        "score_overall": float(score_overall),
        "success": success,
        "parsed_answer": parsed_answer,
        "equation_expr": equation_expr,
        "equation_value": None if equation_value is None else float(equation_value),
        "used_numbers": used_numbers,
        "numbers_match": numbers_match,
        "possible_case_expected": possible_case_expected,
        "cards": cards_list,
        "no_token": no_token,
        "parse_error": parse_error,
        "ground_truth": ground_truth,
    }
