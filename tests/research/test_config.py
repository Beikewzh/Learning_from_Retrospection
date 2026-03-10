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

import pytest

from research.config import ARConfig, LatentConfig


def test_latent_include_prompt_guard():
    cfg = LatentConfig(include_prompt=True)
    with pytest.raises(ValueError, match="include_prompt"):
        cfg.post_init()


def test_ar_window_intervals_must_be_non_negative():
    cfg = ARConfig(window_intervals=-1)
    with pytest.raises(ValueError, match="window_intervals"):
        cfg.post_init()


def test_ar_window_interval_steps_requires_positive_when_set():
    cfg = ARConfig(window_interval_steps=0)
    with pytest.raises(ValueError, match="window_interval_steps"):
        cfg.post_init()


def test_ar_max_age_steps_requires_positive_when_set():
    cfg = ARConfig(max_age_steps=0)
    with pytest.raises(ValueError, match="max_age_steps"):
        cfg.post_init()


def test_ar_start_after_steps_requires_non_negative():
    cfg = ARConfig(start_after_steps=-1)
    with pytest.raises(ValueError, match="start_after_steps"):
        cfg.post_init()


def test_ar_stale_action_must_be_valid():
    cfg = ARConfig(stale_action="panic")
    with pytest.raises(ValueError, match="stale_action"):
        cfg.post_init()
