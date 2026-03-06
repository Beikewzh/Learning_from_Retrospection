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

"""Helpers for extracting and slicing hidden states from model outputs."""

from __future__ import annotations

from typing import Any

import torch


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported latent precision: {precision}")


def cast_latent_dtype(latents: torch.Tensor, precision: str) -> torch.Tensor:
    return latents.to(dtype=precision_to_dtype(precision))


def select_hidden_from_output(output: Any, layer_index: int = -1) -> torch.Tensor:
    """Return hidden state tensor from model output.

    Expected output format from HF CausalLM forward with `output_hidden_states=True`.
    """
    if getattr(output, "hidden_states", None) is None:
        raise RuntimeError("Model output does not contain hidden_states. Enable output_hidden_states=True.")

    hidden_states = output.hidden_states
    num_layers = len(hidden_states)
    if not (-num_layers <= layer_index < num_layers):
        raise IndexError(f"layer_index={layer_index} out of bounds for {num_layers} hidden-state tensors")

    hidden = hidden_states[layer_index]
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected hidden state ndim=3, got {hidden.ndim}")
    return hidden


def slice_latent_tokens(latents: torch.Tensor, response_length: int, include_prompt: bool = False) -> torch.Tensor:
    """Slice per-token latents to either full sequence or response-only window.

    The response-only slice aligns with log-prob window: `[:, -response_len-1:-1, :]`.
    """
    if latents.ndim != 3:
        raise ValueError(f"Expected [batch, seq, hidden], got {tuple(latents.shape)}")
    if include_prompt:
        return latents
    return latents[:, -response_length - 1 : -1, :]
