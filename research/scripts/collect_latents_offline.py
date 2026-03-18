#!/usr/bin/env python3
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

"""Collect response-token latent trajectories from a pretrained reasoning model."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from mathruler.grader import extract_boxed_content, grade_answer
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.buffer import ParquetLatentBuffer
from research.latent.extractor import cast_latent_dtype, select_hidden_from_output, slice_latent_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--input", type=str, required=True, help="Normalized JSONL input path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for buffer + metadata output.")
    parser.add_argument("--prompt-template", type=str, required=True, help="Prompt template file path.")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--id-key", type=str, default="id")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--latent-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples-per-question", type=int, default=1, help="Number of responses to sample per question.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed for generation.")
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
    if cache_root is None:
        return
    root = Path(cache_root).expanduser().resolve()
    hf_home = root / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["XDG_CACHE_HOME"] = str(root)
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


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
        "think_text": None,
        "final_answer_text": None,
        "think_token_start": None,
        "think_token_end": None,
        "answer_token_start": None,
        "answer_token_end": None,
    }

    think_match = re.search(r"<think>(.*?)</think>", normalized, flags=re.DOTALL)
    if think_match is not None:
        prefix = normalized[: think_match.start()]
        think_text = think_match.group(1)
        result["has_think_tags"] = True
        result["think_text"] = think_text
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        think_ids = tokenizer(think_text, add_special_tokens=False)["input_ids"]
        result["think_token_start"] = len(prefix_ids)
        result["think_token_end"] = len(prefix_ids) + len(think_ids)

    answer_match = re.search(r"\\boxed\{(.*)\}", normalized, flags=re.DOTALL)
    if answer_match is not None:
        answer_prefix = normalized[: answer_match.start(1)]
        answer_text = answer_match.group(1)
        result["final_answer_text"] = answer_text
        answer_prefix_ids = tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        result["answer_token_start"] = len(answer_prefix_ids)
        result["answer_token_end"] = len(answer_prefix_ids) + len(answer_ids)

    return result


def score_response(response: str, ground_truth: str) -> dict[str, float]:
    response = normalize_response_text(response)
    format_ok = bool(re.fullmatch(re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL), response))
    answer = extract_boxed_content(response)
    accuracy = 1.0 if grade_answer(answer, ground_truth) else 0.0
    return {
        "format": 1.0 if format_ok else 0.0,
        "accuracy": accuracy,
        "overall": accuracy,
    }


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


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_root)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prompt_template_path = Path(args.prompt_template).expanduser().resolve()
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
    open_mode = "a" if args.resume and metadata_path.exists() else "w"
    since_flush = 0
    with input_path.open(encoding="utf-8") as src, metadata_path.open(open_mode, encoding="utf-8") as meta_out:
        for idx, line in enumerate(src, start=1):
            if args.limit is not None and processed >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row[args.question_key])
            answer = str(row[args.answer_key])
            uid = str(row.get(args.id_key, processed))
            prompt = render_prompt(template, question)
            question_had_new_sample = False
            for sample_idx in range(args.num_samples_per_question):
                sample_uid = uid if args.num_samples_per_question == 1 else f"{uid}::sample_{sample_idx:02d}"
                if sample_uid in existing_uids:
                    skipped_existing += 1
                    continue
                sample_seed = int(args.seed) + processed * max(1, args.num_samples_per_question) + sample_idx

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
                scores = score_response(response=response_text, ground_truth=answer)
                spans = extract_reasoning_spans(response=response_text, tokenizer=tokenizer)
                response_len = torch.tensor([response_ids.size(1)], dtype=torch.long)
                extrinsic_final = torch.tensor([scores["overall"]], dtype=torch.float32)
                success = torch.tensor([scores["accuracy"] > 0.5], dtype=torch.bool)
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

                metadata = {
                    "uid": sample_uid,
                    "question_uid": uid,
                    "sample_idx": sample_idx,
                    "generation_seed": sample_seed,
                    "step": processed,
                    "question": question,
                    "ground_truth": answer,
                    "prompt": prompt,
                    "response": response_text,
                    "normalized_response": spans["normalized_response"],
                    "response_length": int(response_ids.size(1)),
                    "prompt_length": int(full_sequences.size(1) - response_ids.size(1)),
                    "score_overall": scores["overall"],
                    "score_accuracy": scores["accuracy"],
                    "score_format": scores["format"],
                    "success": bool(success.item()),
                    "has_think_tags": spans["has_think_tags"],
                    "think_text": spans["think_text"],
                    "final_answer_text": spans["final_answer_text"],
                    "think_token_start": spans["think_token_start"],
                    "think_token_end": spans["think_token_end"],
                    "answer_token_start": spans["answer_token_start"],
                    "answer_token_end": spans["answer_token_end"],
                    "model": args.model,
                    "layer_index": args.layer_index,
                    "latent_dtype": args.latent_dtype,
                }
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
        "saved_questions": saved_questions,
        "saved_sequences": saved,
        "skipped_existing_sequences": skipped_existing,
        "num_samples_per_question": args.num_samples_per_question,
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
