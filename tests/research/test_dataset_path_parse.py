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

from verl.utils.dataset import parse_dataset_path


def test_parse_plain_dataset_with_split():
    path, config, split = parse_dataset_path("hiyouga/math12k@train")
    assert path == "hiyouga/math12k"
    assert config is None
    assert split == "train"


def test_parse_dataset_with_config_and_split_colon():
    path, config, split = parse_dataset_path("openai/gsm8k:main@test")
    assert path == "openai/gsm8k"
    assert config == "main"
    assert split == "test"


def test_parse_dataset_with_config_and_split_hash():
    path, config, split = parse_dataset_path("openai/gsm8k#socratic@train")
    assert path == "openai/gsm8k"
    assert config == "socratic"
    assert split == "train"
