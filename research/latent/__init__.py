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

"""Latent extraction and serialization helpers."""

from .codec import deserialize_latent_tensor, serialize_latent_tensor
from .extractor import cast_latent_dtype, precision_to_dtype, select_hidden_from_output, slice_latent_tokens

__all__ = [
    "deserialize_latent_tensor",
    "serialize_latent_tensor",
    "cast_latent_dtype",
    "precision_to_dtype",
    "select_hidden_from_output",
    "slice_latent_tokens",
]
