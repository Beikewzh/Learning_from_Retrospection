#!/usr/bin/env python3
"""Collect response-token latent trajectories for Game24 with strict equation scoring."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.buffer import ParquetLatentBuffer
from research.latent.extractor import cast_latent_dtype, select_hidden_from_output, slice_latent_tokens
from research.offline.game24.score_game24 import score_game24_response
from research.scripts.offline_utils import configure_hf_cache, resolve_repo_mounted_path


REQUIRED_METADATA_KEYS = {
    "uid",
    "question_uid",
    "sample_idx",
    "response_length",
    "success",
    "score_accuracy",
    "think_token_start",
    "think_token_end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--input", type=str, required=True, help="Normalized JSONL input path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for buffer + metadata output.")
    parser.add_argument("--prompt-template", type=str, required=True, help="Prompt template file path.")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--id-key", type=str, default="id")
    parser.add_argument("--cards-key", type=str, default="cards")
    parser.add_argument("--is-possible-key", type=str, default="is_possible")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--latent-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples-per-question", type=int, default=1, help="Number of responses to sample per question.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed for generation.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic question shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to process.")
    parser.add_argument("--shard-max-samples", type=int, default=256)
    parser.add_argument(
        "--flush-every-n-samples",
        type=int,
        default=1,
        help="Flush buffer and metadata after this many collected samples. Use 1 for safest preemption recovery.",
    )
    parser.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "none"])
    parser.add_argument("--max-disk-gb", type=float, default=200.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Optional cache root. If set, configures HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE there.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing metadata/buffer in output-dir.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def configure_cache(cache_root: str | None) -> None:
    configure_hf_cache(cache_root)


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, content: str) -> str:
    content = content.strip()
    rendered = re.sub(r"\{\{\s*content\s*\|\s*trim\s*\}\}", lambda _: content, template)
    rendered = re.sub(r"\{\{\s*content\s*\}\}", lambda _: content, rendered)
    if rendered == template:
        raise ValueError("Prompt template does not contain a supported {{ content }} placeholder.")
    return rendered


def normalize_response_text(text: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", text)


def extract_reasoning_spans(response: str, tokenizer: AutoTokenizer) -> dict[str, Any]:
    normalized = normalize_response_text(response)
    result: dict[str, Any] = {
        "normalized_response": normalized,
        "has_think_tags": False,
        "has_answer_tags": False,
        "think_text": None,
        "final_answer_text": None,
        "think_token_start": None,
        "think_token_end": None,
        "answer_token_start": None,
        "answer_token_end": None,
    }

    think_match = re.search(r"<think>(.*?)</think>", normalized, flags=re.DOTALL | re.IGNORECASE)
    if think_match is not None:
        prefix = normalized[: think_match.start()]
        think_text = think_match.group(1)
        result["has_think_tags"] = True
        result["think_text"] = think_text
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        think_ids = tokenizer(think_text, add_special_tokens=False)["input_ids"]
        result["think_token_start"] = len(prefix_ids)
        result["think_token_end"] = len(prefix_ids) + len(think_ids)

    answer_match = re.search(r"<answer>(.*?)</answer>", normalized, flags=re.DOTALL | re.IGNORECASE)
    if answer_match is not None:
        result["has_answer_tags"] = True
        answer_prefix = normalized[: answer_match.start(1)]
        answer_text = answer_match.group(1)
        result["final_answer_text"] = answer_text
        answer_prefix_ids = tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        result["answer_token_start"] = len(answer_prefix_ids)
        result["answer_token_end"] = len(answer_prefix_ids) + len(answer_ids)
    else:
        boxed_match = re.search(r"\\boxed\{(.*)\}", normalized, flags=re.DOTALL)
        if boxed_match is not None:
            answer_prefix = normalized[: boxed_match.start(1)]
            answer_text = boxed_match.group(1)
            result["final_answer_text"] = answer_text
            answer_prefix_ids = tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            result["answer_token_start"] = len(answer_prefix_ids)
            result["answer_token_end"] = len(answer_prefix_ids) + len(answer_ids)

    return result


def generate_one(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_prompt_length: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.size(1)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_response_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        if temperature > 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        full_sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    response_ids = full_sequences[:, prompt_len:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return full_sequences, response_ids, response_text


def extract_response_latents(
    *,
    model: AutoModelForCausalLM,
    full_sequences: torch.Tensor,
    response_ids: torch.Tensor,
    layer_index: int,
    latent_dtype: str,
) -> torch.Tensor:
    attention_mask = torch.ones_like(full_sequences, dtype=torch.long, device=full_sequences.device)
    with torch.no_grad():
        output = model(
            input_ids=full_sequences,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    full_latents = select_hidden_from_output(output=output, layer_index=layer_index)
    latents = slice_latent_tokens(full_latents, response_length=response_ids.size(1), include_prompt=False)
    latents = cast_latent_dtype(latents, precision=latent_dtype)
    return latents


def build_metadata_row(
    *,
    sample_uid: str,
    question_uid: str,
    sample_idx: int,
    sample_seed: int,
    step: int,
    question: str,
    ground_truth: str,
    cards: Any,
    is_possible: Any,
    prompt: str,
    response_text: str,
    spans: dict[str, Any],
    response_len: int,
    prompt_len: int,
    score: dict[str, Any],
    model_name: str,
    layer_index: int,
    latent_dtype: str,
) -> dict[str, Any]:
    metadata = {
        "uid": sample_uid,
        "question_uid": question_uid,
        "sample_idx": sample_idx,
        "generation_seed": sample_seed,
        "step": step,
        "question": question,
        "ground_truth": ground_truth,
        "cards": cards,
        "is_possible": score["possible_case_expected"] if "possible_case_expected" in score else is_possible,
        "prompt": prompt,
        "response": response_text,
        "normalized_response": spans["normalized_response"],
        "response_length": int(response_len),
        "prompt_length": int(prompt_len),
        "score_overall": score["score_overall"],
        "score_accuracy": score["score_accuracy"],
        "score_format": score["score_format"],
        "success": bool(score["success"]),
        "has_think_tags": spans["has_think_tags"],
        "has_answer_tags": spans["has_answer_tags"],
        "think_text": spans["think_text"],
        "final_answer_text": spans["final_answer_text"],
        "think_token_start": spans["think_token_start"],
        "think_token_end": spans["think_token_end"],
        "answer_token_start": spans["answer_token_start"],
        "answer_token_end": spans["answer_token_end"],
        "parsed_answer": score.get("parsed_answer"),
        "equation_expr": score.get("equation_expr"),
        "equation_value": score.get("equation_value"),
        "used_numbers": score.get("used_numbers"),
        "numbers_match": score.get("numbers_match"),
        "no_token": score.get("no_token"),
        "parse_error": score.get("parse_error"),
        "model": model_name,
        "layer_index": layer_index,
        "latent_dtype": latent_dtype,
    }
    missing = REQUIRED_METADATA_KEYS.difference(metadata)
    if missing:
        raise KeyError(f"Metadata missing required keys: {sorted(missing)}")
    return metadata


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    configure_cache(args.cache_root)
    input_path = resolve_repo_mounted_path(args.input)
    output_dir = resolve_repo_mounted_path(args.output_dir)
    prompt_template_path = resolve_repo_mounted_path(args.prompt_template)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "collection_summary.json"
    buffer_dir = output_dir / "buffer"

    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    existing_uids: set[str] = set()
    if args.overwrite:
        if metadata_path.exists():
            metadata_path.unlink()
        if summary_path.exists():
            summary_path.unlink()
        if buffer_dir.exists():
            shutil.rmtree(buffer_dir)
    elif args.resume:
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    existing_uids.add(str(row["uid"]))
    elif metadata_path.exists():
        raise FileExistsError(f"{metadata_path} already exists. Use --overwrite or choose a new output dir.")

    device = torch.device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    print("Cache configuration:")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME"]:
        print(f"  {key}={os.environ.get(key, '<unset>')}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)
    model.eval()

    template = load_prompt_template(prompt_template_path)
    buffer = ParquetLatentBuffer(
        root_dir=str(buffer_dir),
        shard_max_samples=args.shard_max_samples,
        compression=args.compression,
        max_disk_gb=args.max_disk_gb,
    )

    processed = 0
    saved = len(existing_uids)
    saved_questions = 0
    skipped_existing = 0
    seen_questions = 0
    selected_questions = 0
    open_mode = "a" if args.resume and metadata_path.exists() else "w"
    since_flush = 0

    with input_path.open(encoding="utf-8") as src, metadata_path.open(open_mode, encoding="utf-8") as meta_out:
        for line in src:
            if not line.strip():
                continue
            if args.limit is not None and seen_questions >= args.limit:
                break
            question_idx = seen_questions
            seen_questions += 1
            if question_idx % args.num_shards != args.shard_index:
                continue

            row = json.loads(line)
            question = str(row[args.question_key])
            answer = str(row[args.answer_key])
            cards = row[args.cards_key]
            is_possible = row[args.is_possible_key]
            uid = str(row.get(args.id_key, processed))
            prompt = render_prompt(template, question)
            question_had_new_sample = False

            for sample_idx in range(args.num_samples_per_question):
                sample_uid = uid if args.num_samples_per_question == 1 else f"{uid}::sample_{sample_idx:02d}"
                if sample_uid in existing_uids:
                    skipped_existing += 1
                    continue

                sample_seed = int(args.seed) + question_idx * max(1, args.num_samples_per_question) + sample_idx

                full_sequences, response_ids, response_text = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_prompt_length=args.max_prompt_length,
                    max_response_length=args.max_response_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=sample_seed,
                    device=device,
                )

                latents = extract_response_latents(
                    model=model,
                    full_sequences=full_sequences,
                    response_ids=response_ids,
                    layer_index=args.layer_index,
                    latent_dtype=args.latent_dtype,
                )

                score = score_game24_response(
                    response=response_text,
                    cards=cards,
                    is_possible=is_possible,
                    ground_truth=answer,
                )
                spans = extract_reasoning_spans(response=response_text, tokenizer=tokenizer)

                response_len = torch.tensor([response_ids.size(1)], dtype=torch.long)
                extrinsic_final = torch.tensor([score["score_overall"]], dtype=torch.float32)
                success = torch.tensor([score["success"]], dtype=torch.bool)

                buffer.append_batch(
                    uids=[sample_uid],
                    step=processed,
                    latents=latents.cpu(),
                    response_lengths=response_len,
                    extrinsic_final=extrinsic_final,
                    success=success,
                    dtype=args.latent_dtype,
                )
                buffer.flush()

                metadata = build_metadata_row(
                    sample_uid=sample_uid,
                    question_uid=uid,
                    sample_idx=sample_idx,
                    sample_seed=sample_seed,
                    step=processed,
                    question=question,
                    ground_truth=answer,
                    cards=cards,
                    is_possible=is_possible,
                    prompt=prompt,
                    response_text=response_text,
                    spans=spans,
                    response_len=int(response_ids.size(1)),
                    prompt_len=int(full_sequences.size(1) - response_ids.size(1)),
                    score=score,
                    model_name=args.model,
                    layer_index=args.layer_index,
                    latent_dtype=args.latent_dtype,
                )

                meta_out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                meta_out.flush()

                existing_uids.add(sample_uid)
                question_had_new_sample = True
                if response_ids.size(1) > 1:
                    saved += 1
                since_flush += 1
                if since_flush >= max(1, args.flush_every_n_samples):
                    meta_out.flush()
                    since_flush = 0

            processed += 1
            selected_questions += 1
            if question_had_new_sample:
                saved_questions += 1

    buffer.flush()
    summary = {
        "model": args.model,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "buffer_dir": str(buffer_dir),
        "metadata_path": str(metadata_path),
        "processed_examples": processed,
        "seen_questions_before_shard_filter": seen_questions,
        "selected_questions": selected_questions,
        "saved_questions": saved_questions,
        "saved_sequences": saved,
        "skipped_existing_sequences": skipped_existing,
        "num_samples_per_question": args.num_samples_per_question,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "temperature": args.temperature,
        "seed": args.seed,
        "resume": bool(args.resume),
        "layer_index": args.layer_index,
        "latent_dtype": args.latent_dtype,
        "device": args.device,
        "buffer_stats": buffer.get_stats(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
