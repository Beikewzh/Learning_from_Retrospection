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

"""Intrinsic reward transformation utilities."""

from .reward_rule import (
    INTRINSIC_GATE_REGISTRY,
    RunningZScore,
    apply_intrinsic_rule,
    compose_total_advantage,
    compute_intrinsic_gate,
)

__all__ = [
    "RunningZScore",
    "apply_intrinsic_rule",
    "compute_intrinsic_gate",
    "compose_total_advantage",
    "INTRINSIC_GATE_REGISTRY",
]
