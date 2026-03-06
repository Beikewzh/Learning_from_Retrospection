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

import torch

from research.latent.codec import deserialize_latent_tensor, serialize_latent_tensor


def test_codec_roundtrip_fp16():
    x = torch.randn(7, 11, dtype=torch.float32)
    blob, seq_len, hidden_dim, dtype = serialize_latent_tensor(x, dtype="fp16")
    y = deserialize_latent_tensor(blob, seq_len=seq_len, hidden_dim=hidden_dim, dtype=dtype)

    assert y.shape == x.shape
    assert y.dtype == torch.float16
    assert torch.allclose(y.float(), x, atol=1e-2, rtol=1e-2)


def test_codec_roundtrip_fp32():
    x = torch.randn(5, 3, dtype=torch.float32)
    blob, seq_len, hidden_dim, dtype = serialize_latent_tensor(x, dtype="fp32")
    y = deserialize_latent_tensor(blob, seq_len=seq_len, hidden_dim=hidden_dim, dtype=dtype)

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)
