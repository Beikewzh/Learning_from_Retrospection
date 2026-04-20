#!/usr/bin/env python3
"""Prepare normalized Game24 JSONL data for the offline latent collector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CARDS_PATTERN = re.compile(r"Cards\s*:\s*([^\.\n]+)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to source JSON/JSONL file (e.g. all_24_game_results_shuffled.json).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to normalized output JSONL.",
    )
    parser.add_argument("--id-key", type=str, default="id", help="Input key for id (fallback to row index).")
    parser.add_argument("--question-key", type=str, default="question", help="Input key for question text.")
    parser.add_argument("--answer-key", type=str, default="answer", help="Input key for answer text.")
    parser.add_argument("--cards-key", type=str, default="cards", help="Input key for cards (optional if derivable from question).")
    parser.add_argument(
        "--is-possible-key",
        type=str,
        default="is_possible",
        help="Input key for solvability label.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to write.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as src:
            for idx, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError(f"Line {idx} is not a JSON object.")
                rows.append(obj)
        return rows

    if isinstance(parsed, list):
        if not all(isinstance(x, dict) for x in parsed):
            raise TypeError("Top-level JSON list must contain only objects.")
        return parsed
    if isinstance(parsed, dict):
        for key in ("data", "rows", "items", "examples"):
            value = parsed.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value
    raise TypeError("Unsupported input JSON shape. Expected list[object], JSONL, or dict with data list.")


def parse_cards(cards_raw: Any) -> list[int]:
    if isinstance(cards_raw, (list, tuple)):
        cards = [int(x) for x in cards_raw]
    elif isinstance(cards_raw, str):
        nums = re.findall(r"-?\d+", cards_raw)
        cards = [int(x) for x in nums]
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
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def derive_cards_from_text(text: str) -> list[int] | None:
    match = CARDS_PATTERN.search(text)
    if match is None:
        return None
    return parse_cards(match.group(1))


def normalize_row(
    row: dict[str, Any],
    *,
    row_index: int,
    id_key: str,
    question_key: str,
    answer_key: str,
    cards_key: str,
    is_possible_key: str,
) -> dict[str, Any]:
    if question_key not in row:
        raise KeyError(f"Missing question key '{question_key}' at row {row_index}")
    if answer_key not in row:
        raise KeyError(f"Missing answer key '{answer_key}' at row {row_index}")
    if is_possible_key not in row:
        raise KeyError(f"Missing is_possible key '{is_possible_key}' at row {row_index}")

    question = str(row[question_key]).strip()
    answer = str(row[answer_key]).strip()
    if not question:
        raise ValueError(f"Empty question at row {row_index}")
    if not answer:
        raise ValueError(f"Empty answer at row {row_index}")

    cards_raw = row.get(cards_key)
    cards: list[int]
    if cards_raw is None:
        cards = derive_cards_from_text(question) or derive_cards_from_text(str(row.get("prompt", "")))
        if cards is None:
            raise KeyError(
                f"Missing cards key '{cards_key}' and could not derive cards from question/prompt at row {row_index}"
            )
    else:
        cards = parse_cards(cards_raw)

    is_possible = parse_boolish(row[is_possible_key])

    normalized = {
        "id": str(row.get(id_key, row_index)),
        "question": question,
        "answer": answer,
        "cards": ", ".join(str(x) for x in cards),
        "is_possible": is_possible,
    }

    extras = {
        key: value
        for key, value in row.items()
        if key not in {id_key, question_key, answer_key, cards_key, is_possible_key}
    }
    if extras:
        normalized["meta"] = extras
    return normalized


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    written = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for row_index, row in enumerate(rows):
            if args.limit is not None and written >= args.limit:
                break
            normalized = normalize_row(
                row,
                row_index=row_index,
                id_key=args.id_key,
                question_key=args.question_key,
                answer_key=args.answer_key,
                cards_key=args.cards_key,
                is_possible_key=args.is_possible_key,
            )
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} normalized rows to {output_path}")


if __name__ == "__main__":
    main()
