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

from verl.trainer.config import DataConfig


def test_dataset_preset_math12k_math500_fills_empty_fields():
    cfg = DataConfig(
        dataset_preset="math12k_math500",
        train_files="",
        val_files="",
        prompt_key="",
        answer_key="",
        format_prompt=None,
    )
    cfg.post_init()
    assert cfg.train_files == "hiyouga/math12k@train"
    assert cfg.val_files == "HuggingFaceH4/MATH-500@test"
    assert cfg.prompt_key == "problem"
    assert cfg.answer_key == "answer"
    assert cfg.format_prompt is not None


def test_dataset_preset_does_not_override_explicit_values():
    cfg = DataConfig(
        dataset_preset="math12k_math500",
        train_files="custom/train@train",
        val_files="custom/val@test",
        prompt_key="question",
        answer_key="gt",
    )
    cfg.post_init()
    assert cfg.train_files == "custom/train@train"
    assert cfg.val_files == "custom/val@test"
    assert cfg.prompt_key == "question"
    assert cfg.answer_key == "gt"
