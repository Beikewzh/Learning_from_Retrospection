import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from mathruler.grader import extract_boxed_content, grade_answer
from transformers import AutoModel, AutoTokenizer


REWARD_NAME = "math_prm_step_reward"
REWARD_TYPE = "batch"

_PRM_CACHE: dict[str, Any] = {}
_STEP_SEPARATOR = "<extra_0>"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_MULTISPACE_TAG_RE = re.compile(r"\s*(<|>|/)\s*")
_STEP_SPLIT_RE = re.compile(r"\n\s*\n+")


def _normalize_response_text(response: str) -> str:
    return re.sub(_MULTISPACE_TAG_RE, r"\1", response).strip()


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _extract_reasoning_steps(response: str) -> list[str]:
    match = _THINK_RE.search(response)
    if not match:
        return []

    reasoning = match.group(1).strip()
    if not reasoning:
        return []

    steps = [step.strip() for step in _STEP_SPLIT_RE.split(reasoning) if step.strip()]
    return steps


def _append_final_answer_to_last_step(steps: list[str], response: str) -> list[str]:
    answer = extract_boxed_content(response)
    if not steps or not answer:
        return steps

    boxed = f"Final answer: \\boxed{{{answer}}}."
    steps = list(steps)
    if boxed not in steps[-1]:
        steps[-1] = f"{steps[-1]}\n\n{boxed}"
    return steps


def _get_prm_components(
    model_name: str,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
):
    cache_key = f"{model_name}|{trust_remote_code}|{torch_dtype}|{device_map}"
    cached = _PRM_CACHE.get(cache_key)
    if cached is not None:
        return cached["tokenizer"], cached["model"], cached["step_sep_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    dtype = getattr(torch, torch_dtype)
    model = AutoModel.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).eval()

    step_sep_tokens = tokenizer.encode(_STEP_SEPARATOR, add_special_tokens=False)
    if len(step_sep_tokens) != 1:
        raise ValueError(f"{_STEP_SEPARATOR} must tokenize to one token, got {step_sep_tokens}.")
    step_sep_id = step_sep_tokens[0]

    _PRM_CACHE[cache_key] = {"tokenizer": tokenizer, "model": model, "step_sep_id": step_sep_id}
    return tokenizer, model, step_sep_id


def _build_prm_conversation(problem: str, steps: list[str], tokenizer) -> str:
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": _STEP_SEPARATOR.join(steps) + _STEP_SEPARATOR},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _score_step_rewards(
    problems: list[str],
    step_lists: list[list[str]],
    model_name: str,
    max_length: int = 4096,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> list[list[float]]:
    tokenizer, model, step_sep_id = _get_prm_components(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    conversations = [_build_prm_conversation(problem, steps, tokenizer) for problem, steps in zip(problems, step_lists)]
    inputs = tokenizer(
        conversations,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "logits", outputs[0])
    probs = F.softmax(logits, dim=-1)
    token_masks = inputs["input_ids"] == step_sep_id

    all_scores: list[list[float]] = []
    for batch_idx in range(probs.size(0)):
        step_probs = probs[batch_idx, token_masks[batch_idx], 1]
        all_scores.append(step_probs.detach().cpu().tolist())
    return all_scores


def compute_score(
    reward_inputs: list[dict[str, Any]],
    prm_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B",
    prm_weight: float = 0.1,
    format_weight: float = 0.1,
    max_prm_length: int = 4096,
    max_prm_steps: int = 8,
    min_step_chars: int = 8,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> list[dict[str, float]]:
    normalized_responses = [_normalize_response_text(reward_input["response"]) for reward_input in reward_inputs]
    problems: list[str] = []
    valid_step_lists: list[list[str]] = []
    valid_indices: list[int] = []

    for idx, (reward_input, response) in enumerate(zip(reward_inputs, normalized_responses)):
        steps = [step for step in _extract_reasoning_steps(response) if len(step) >= min_step_chars]
        if not steps:
            continue

        steps = _append_final_answer_to_last_step(steps, response)[:max_prm_steps]
        if not steps:
            continue

        problem = str(reward_input.get("source_prompt") or reward_input.get("prompt") or "").strip()
        problems.append(problem)
        valid_step_lists.append(steps)
        valid_indices.append(idx)

    prm_scores_by_index: dict[int, list[float]] = {}
    if valid_indices:
        all_step_scores = _score_step_rewards(
            problems=problems,
            step_lists=valid_step_lists,
            model_name=prm_model_path,
            max_length=max_prm_length,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        prm_scores_by_index = dict(zip(valid_indices, all_step_scores))

    scores = []
    for idx, (reward_input, response) in enumerate(zip(reward_inputs, normalized_responses)):
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        raw_score = (1 - format_weight) * accuracy_score + format_weight * format_score

        step_scores = prm_scores_by_index.get(idx, [])
        prm_process_product = math.prod(step_scores) if step_scores else 0.0
        prm_process_min = min(step_scores) if step_scores else 0.0
        prm_process_last = step_scores[-1] if step_scores else 0.0
        prm_process_mean = sum(step_scores) / len(step_scores) if step_scores else 0.0

        scores.append(
            {
                "overall": raw_score + prm_weight * prm_process_product,
                "accuracy": accuracy_score,
                "format": format_score,
                "raw_score": raw_score,
                "prm_process_product": prm_process_product,
                "prm_process_min": prm_process_min,
                "prm_process_last": prm_process_last,
                "prm_process_mean": prm_process_mean,
                "num_prm_steps": float(len(step_scores)),
            }
        )

    return scores
