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

"""Byte codec utilities for variable-length latent tensors."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _dtype_str_to_torch(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported latent dtype: {dtype}")


def _dtype_str_to_numpy(dtype: str) -> np.dtype:
    if dtype == "fp16":
        return np.float16
    if dtype == "fp32":
        return np.float32
    raise ValueError(f"Unsupported latent dtype: {dtype}")


def serialize_latent_tensor(tensor: torch.Tensor, dtype: str = "fp16") -> Tuple[bytes, int, int, str]:
    """Serialize a 2D latent tensor [seq_len, hidden_dim] to bytes."""
    if tensor.ndim != 2:
        raise ValueError(f"Expected [seq_len, hidden_dim], got shape={tuple(tensor.shape)}")

    target_dtype = _dtype_str_to_torch(dtype)
    arr = tensor.detach().to(device="cpu", dtype=target_dtype).contiguous().numpy()
    seq_len, hidden_dim = arr.shape
    return arr.tobytes(), int(seq_len), int(hidden_dim), dtype


def deserialize_latent_tensor(blob: bytes, seq_len: int, hidden_dim: int, dtype: str = "fp16") -> torch.Tensor:
    """Deserialize latent bytes into a torch tensor [seq_len, hidden_dim]."""
    np_dtype = _dtype_str_to_numpy(dtype)
    arr = np.frombuffer(blob, dtype=np_dtype).copy()
    expected = seq_len * hidden_dim
    if arr.size != expected:
        raise ValueError(f"Latent blob size mismatch: got {arr.size}, expected {expected}")
    arr = arr.reshape(seq_len, hidden_dim)
    return torch.from_numpy(arr)
