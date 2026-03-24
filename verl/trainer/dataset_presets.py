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

"""Dataset preset registry utilities for training/evaluation portability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetPreset:
    train_files: str
    val_files: str
    prompt_key: str
    answer_key: str
    format_prompt: Optional[str] = None


DATASET_PRESET_REGISTRY: dict[str, DatasetPreset] = {
    "none": DatasetPreset(
        train_files="",
        val_files="",
        prompt_key="prompt",
        answer_key="answer",
        format_prompt=None,
    ),
    "math12k_math500": DatasetPreset(
        train_files="hiyouga/math12k@train",
        val_files="HuggingFaceH4/MATH-500@test",
        prompt_key="problem",
        answer_key="answer",
        format_prompt="./examples/format_prompt/math.jinja",
    ),
}


def apply_dataset_preset(config) -> None:
    """Apply dataset preset defaults to a DataConfig-like object.

    Only fills unset/empty fields so explicit overrides remain in control.
    """
    preset_name = getattr(config, "dataset_preset", "none")
    if not preset_name or preset_name == "none":
        return
    if preset_name not in DATASET_PRESET_REGISTRY:
        supported = ", ".join(sorted(DATASET_PRESET_REGISTRY.keys()))
        raise ValueError(f"Unknown data.dataset_preset: {preset_name}. Supported: {supported}.")

    preset = DATASET_PRESET_REGISTRY[preset_name]
    if not getattr(config, "train_files", ""):
        config.train_files = preset.train_files
    if not getattr(config, "val_files", ""):
        config.val_files = preset.val_files
    if getattr(config, "prompt_key", "") in {"", "prompt"}:
        config.prompt_key = preset.prompt_key
    if getattr(config, "answer_key", "") in {"", "answer"}:
        config.answer_key = preset.answer_key
    if not getattr(config, "format_prompt", None):
        config.format_prompt = preset.format_prompt
