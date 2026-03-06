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

from research.latent.extractor import cast_latent_dtype, select_hidden_from_output, slice_latent_tokens


class _Output:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


def test_select_and_slice_response_latents():
    hs0 = torch.randn(2, 8, 4)
    hs1 = torch.randn(2, 8, 4)
    output = _Output(hidden_states=(hs0, hs1))

    selected = select_hidden_from_output(output, layer_index=-1)
    sliced = slice_latent_tokens(selected, response_length=3, include_prompt=False)

    assert selected.shape == (2, 8, 4)
    assert sliced.shape == (2, 3, 4)
    assert torch.allclose(sliced, selected[:, -4:-1, :])


def test_cast_dtype():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    y = cast_latent_dtype(x, precision="fp16")
    assert y.dtype == torch.float16
