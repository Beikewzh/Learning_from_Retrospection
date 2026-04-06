from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research.offline.game24.collect_latents_game24_offline import (
    REQUIRED_METADATA_KEYS,
    build_metadata_row,
)
from research.offline.game24.prepare_game24_jsonl import normalize_row
from research.offline.game24.score_game24 import score_game24_response


def test_score_game24_valid_solvable_expression() -> None:
    result = score_game24_response(
        response="<think>Try one construction.</think><answer>8*(5-(6-4))</answer>",
        cards="4, 5, 6, 8",
        is_possible=True,
        ground_truth="8*(5-(6-4))=24",
    )
    assert result["score_accuracy"] == 1.0
    assert result["score_format"] == 1.0
    assert result["success"] is True
    assert result["numbers_match"] is True
    assert result["equation_value"] == 24.0


def test_score_game24_wrong_value() -> None:
    result = score_game24_response(
        response="<think>Test value mismatch.</think><answer>(8+4)+(6+5)</answer>",
        cards=[4, 5, 6, 8],
        is_possible=True,
        ground_truth=None,
    )
    assert result["score_accuracy"] == 0.0
    assert result["numbers_match"] is True
    assert result["equation_value"] == 23.0


def test_score_game24_wrong_number_usage() -> None:
    result = score_game24_response(
        response="<think>Uses one number twice.</think><answer>(8+4)+(6+6)</answer>",
        cards="4,5,6,8",
        is_possible=True,
        ground_truth=None,
    )
    assert result["score_accuracy"] == 0.0
    assert result["numbers_match"] is False
    assert result["success"] is False


def test_score_game24_unsolvable_with_correct_no() -> None:
    result = score_game24_response(
        response="<think>I checked all compositions.</think><answer>NO</answer>",
        cards=[3, 5, 7, 7],
        is_possible=False,
        ground_truth="NO",
    )
    assert result["score_accuracy"] == 1.0
    assert result["success"] is True
    assert result["no_token"] is True


def test_score_game24_unsolvable_with_non_no() -> None:
    result = score_game24_response(
        response="<think>Attempting an invalid expression.</think><answer>(7+7)+(5+3)</answer>",
        cards=[3, 5, 7, 7],
        is_possible=False,
        ground_truth="NO",
    )
    assert result["score_accuracy"] == 0.0
    assert result["success"] is False
    assert result["no_token"] is False


def test_prepare_game24_normalized_schema() -> None:
    raw_row = {
        "example_id": "q_001",
        "question": "Find 24.\nCards:3, 5, 7, 7.",
        "answer": "NO",
        "is_possible": False,
        "source": "24-Game-Reasoning",
    }

    normalized = normalize_row(
        raw_row,
        row_index=0,
        id_key="example_id",
        question_key="question",
        answer_key="answer",
        cards_key="cards",
        is_possible_key="is_possible",
    )

    assert set(normalized.keys()) == {"id", "question", "answer", "cards", "is_possible", "meta"}
    assert normalized["id"] == "q_001"
    assert isinstance(normalized["question"], str)
    assert isinstance(normalized["answer"], str)
    assert normalized["cards"] == "3, 5, 7, 7"
    assert isinstance(normalized["is_possible"], bool)
    assert normalized["meta"]["source"] == "24-Game-Reasoning"


def test_metadata_row_contains_required_downstream_keys() -> None:
    spans = {
        "normalized_response": "<think>t</think><answer>NO</answer>",
        "has_think_tags": True,
        "has_answer_tags": True,
        "think_text": "t",
        "final_answer_text": "NO",
        "think_token_start": 0,
        "think_token_end": 1,
        "answer_token_start": 2,
        "answer_token_end": 3,
    }
    score = {
        "score_overall": 1.0,
        "score_accuracy": 1.0,
        "score_format": 1.0,
        "success": True,
        "parsed_answer": "NO",
        "equation_expr": None,
        "equation_value": None,
        "used_numbers": None,
        "numbers_match": None,
        "no_token": True,
        "parse_error": None,
        "possible_case_expected": False,
    }

    row = build_metadata_row(
        sample_uid="uid_0",
        question_uid="q_0",
        sample_idx=0,
        sample_seed=7,
        step=0,
        question="Cards:3,5,7,7",
        ground_truth="NO",
        cards="3,5,7,7",
        is_possible=False,
        prompt="prompt",
        response_text="<think>t</think><answer>NO</answer>",
        spans=spans,
        response_len=3,
        prompt_len=12,
        score=score,
        model_name="test/model",
        layer_index=-1,
        latent_dtype="fp16",
    )

    assert REQUIRED_METADATA_KEYS.issubset(row.keys())
    assert row["uid"] == "uid_0"
    assert row["question_uid"] == "q_0"
    assert isinstance(row["response_length"], int)
    assert isinstance(row["success"], bool)
    assert isinstance(row["score_accuracy"], float)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_game24_pipeline_smoke_merge_and_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "offline_game24" / "toy_model_limit20"

    shard0 = run_root / "shard_00"
    shard1 = run_root / "shard_01"
    shard0.mkdir(parents=True)
    shard1.mkdir(parents=True)

    _write_jsonl(
        shard0 / "metadata.jsonl",
        [
            {"uid": "q0::sample_00", "question_uid": "q0", "step": 0},
            {"uid": "q1::sample_00", "question_uid": "q1", "step": 1},
        ],
    )
    _write_jsonl(
        shard1 / "metadata.jsonl",
        [
            {"uid": "q2::sample_00", "question_uid": "q2", "step": 0},
            {"uid": "q1::sample_00", "question_uid": "q1", "step": 1},
        ],
    )

    (shard0 / "buffer" / "shards").mkdir(parents=True)
    (shard1 / "buffer" / "shards").mkdir(parents=True)
    (shard0 / "buffer" / "shards" / "part-000000.parquet").write_bytes(b"PAR1")
    (shard1 / "buffer" / "shards" / "part-000000.parquet").write_bytes(b"PAR1")

    (shard0 / "collection_summary.json").write_text(
        json.dumps({"buffer_stats": {"buffer/total_samples": 2}}),
        encoding="utf-8",
    )
    (shard1 / "collection_summary.json").write_text(
        json.dumps({"buffer_stats": {"buffer/total_samples": 2}}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.offline.game24.merge_game24_shards",
            "--input-root",
            str(run_root),
            "--overwrite",
        ],
        check=True,
    )

    merged_run = run_root / "merged"
    assert (merged_run / "metadata.jsonl").exists()
    assert (merged_run / "collection_summary.json").exists()
    assert (merged_run / "buffer" / "manifest.json").exists()
    assert len(list((merged_run / "buffer" / "shards").glob("part-*.parquet"))) == 2

    analysis_root = merged_run / "analysis_parallel"
    ar_model_ckpt = analysis_root / "ar_model" / "checkpoints" / "latest.pt"
    ar_model_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ar_model_ckpt.write_bytes(b"fake checkpoint")

    shards_dir = analysis_root / "shards"
    _write_jsonl(
        shards_dir / "metrics_shard_00.jsonl",
        [
            {
                "uid": "q0::sample_00",
                "question_uid": "q0",
                "success": True,
                "score_accuracy": 1.0,
                "response_length": 45,
                "analysis_length": 40,
                "decay_rate": 0.62,
                "ar_error": 0.09,
            },
            {
                "uid": "q1::sample_00",
                "question_uid": "q1",
                "success": False,
                "score_accuracy": 0.0,
                "response_length": 39,
                "analysis_length": 35,
                "decay_rate": 0.71,
                "ar_error": 0.19,
            },
        ],
    )
    _write_jsonl(
        shards_dir / "metrics_shard_01.jsonl",
        [
            {
                "uid": "q2::sample_00",
                "question_uid": "q2",
                "success": True,
                "score_accuracy": 1.0,
                "response_length": 28,
                "analysis_length": 24,
                "decay_rate": 0.55,
                "ar_error": 0.05,
            },
            {
                "uid": "q1::sample_00",
                "question_uid": "q1",
                "success": False,
                "score_accuracy": 0.0,
                "response_length": 39,
                "analysis_length": 35,
                "decay_rate": 0.71,
                "ar_error": 0.19,
            },
        ],
    )
    (shards_dir / "metrics_shard_00_summary.json").write_text("{}", encoding="utf-8")
    (shards_dir / "metrics_shard_01_summary.json").write_text("{}", encoding="utf-8")

    merged_metrics_dir = analysis_root / "merged"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.offline.game24.merge_game24_analysis_metrics",
            "--input-dir",
            str(shards_dir),
            "--output-dir",
            str(merged_metrics_dir),
            "--overwrite",
        ],
        check=True,
    )

    assert (analysis_root / "ar_model" / "checkpoints" / "latest.pt").exists()
    assert (analysis_root / "shards" / "metrics_shard_00.jsonl").exists()
    assert (analysis_root / "shards" / "metrics_shard_01.jsonl").exists()
    assert (analysis_root / "merged" / "metrics.jsonl").exists()

    rows = [json.loads(line) for line in (merged_metrics_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    uids = [row["uid"] for row in rows]
    assert len(uids) == len(set(uids))
    assert len(rows) == 3


def test_merge_game24_shards_supports_single_shard_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "single_layout"
    (run_root / "buffer" / "shards").mkdir(parents=True)
    _write_jsonl(run_root / "metadata.jsonl", [{"uid": "u0", "question_uid": "q0", "step": 0}])
    (run_root / "buffer" / "shards" / "part-000000.parquet").write_bytes(b"PAR1")
    (run_root / "collection_summary.json").write_text(
        json.dumps({"buffer_stats": {"buffer/total_samples": 1}}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.offline.game24.merge_game24_shards",
            "--input-root",
            str(run_root),
            "--overwrite",
        ],
        check=True,
    )
    assert (run_root / "merged" / "metadata.jsonl").exists()
