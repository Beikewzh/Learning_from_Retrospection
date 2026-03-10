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

from types import SimpleNamespace

import torch
import pytest

from research.config import ResearchConfig
from research.manager import LeaRSManager


def _append_one_sample(manager: LeaRSManager, step: int) -> None:
    latents = torch.randn(1, 4, 4)
    response_lengths = torch.tensor([4])
    extrinsic = torch.tensor([1.0])
    success = torch.tensor([True])
    manager.buffer.append_batch(
        uids=[f"uid-{step}"],
        step=step,
        latents=latents,
        response_lengths=response_lengths,
        extrinsic_final=extrinsic,
        success=success,
        dtype="fp16",
    )


def test_maybe_train_ar_uses_recent_window(tmp_path):
    cfg = ResearchConfig(enabled=True)
    cfg.buffer.shard_max_samples = 64
    cfg.ar.min_buffer_samples = 1
    cfg.ar.train_every_n_steps = 10
    cfg.ar.window_intervals = 1
    cfg.ar.window_interval_steps = 10
    cfg.post_init()

    manager = LeaRSManager(config=cfg, checkpoint_root=str(tmp_path))
    _append_one_sample(manager, step=10)
    _append_one_sample(manager, step=20)
    _append_one_sample(manager, step=30)
    manager.buffer.flush()

    captured = {}

    def fake_train(sequences, global_step):
        captured["samples"] = len(sequences)
        captured["global_step"] = global_step
        return SimpleNamespace(train_loss=0.1, steps=1)

    manager.ar_trainer.train_from_sequences = fake_train

    metrics = manager.maybe_train_ar(global_step=30)
    manager.close()

    assert metrics["research/ar/trained"] == 1.0
    assert captured["global_step"] == 30
    assert captured["samples"] == 1
    assert metrics["research/ar/window_min_step"] == 21.0
    assert metrics["research/ar/windowed_samples"] == 1.0


def test_maybe_train_ar_window_zero_uses_all_rows(tmp_path):
    cfg = ResearchConfig(enabled=True)
    cfg.buffer.shard_max_samples = 64
    cfg.ar.min_buffer_samples = 1
    cfg.ar.train_every_n_steps = 10
    cfg.ar.window_intervals = 0
    cfg.post_init()

    manager = LeaRSManager(config=cfg, checkpoint_root=str(tmp_path))
    _append_one_sample(manager, step=10)
    _append_one_sample(manager, step=20)
    _append_one_sample(manager, step=30)
    manager.buffer.flush()

    captured = {}

    def fake_train(sequences, global_step):
        captured["samples"] = len(sequences)
        captured["global_step"] = global_step
        return SimpleNamespace(train_loss=0.1, steps=1)

    manager.ar_trainer.train_from_sequences = fake_train

    metrics = manager.maybe_train_ar(global_step=30)
    manager.close()

    assert metrics["research/ar/trained"] == 1.0
    assert captured["global_step"] == 30
    assert captured["samples"] == 3
    assert metrics["research/ar/window_min_step"] == 0.0
    assert metrics["research/ar/windowed_samples"] == 3.0


def _fake_batch(latents: torch.Tensor, response_mask: torch.Tensor):
    return SimpleNamespace(batch={"latent_response_last": latents, "response_mask": response_mask})


def test_compute_intrinsic_emits_age_steps_and_warns_when_stale(tmp_path):
    cfg = ResearchConfig(enabled=True)
    cfg.ar.max_age_steps = 2
    cfg.ar.stale_action = "warn"
    cfg.post_init()

    manager = LeaRSManager(config=cfg, checkpoint_root=str(tmp_path))
    manager._scorer = SimpleNamespace(
        meta=SimpleNamespace(global_step=1, train_loss=0.5),
        score=lambda latents, response_mask: torch.ones_like(response_mask, dtype=torch.float32),
    )

    latents = torch.randn(1, 4, 4)
    response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    batch = _fake_batch(latents=latents, response_mask=response_mask)
    extrinsic_scores = torch.zeros_like(response_mask, dtype=torch.float32)

    with pytest.warns(UserWarning, match="stale"):
        _, metrics = manager.compute_intrinsic(batch=batch, extrinsic_scores=extrinsic_scores, global_step=5)
    manager.close()

    assert metrics["research/intrinsic/active"] == 1.0
    assert metrics["research/ar/age_steps"] == 4.0
    assert metrics["research/ar/stale"] == 1.0


def test_compute_intrinsic_fails_when_stale_action_fail(tmp_path):
    cfg = ResearchConfig(enabled=True)
    cfg.ar.max_age_steps = 1
    cfg.ar.stale_action = "fail"
    cfg.post_init()

    manager = LeaRSManager(config=cfg, checkpoint_root=str(tmp_path))
    manager._scorer = SimpleNamespace(
        meta=SimpleNamespace(global_step=1, train_loss=0.5),
        score=lambda latents, response_mask: torch.ones_like(response_mask, dtype=torch.float32),
    )

    latents = torch.randn(1, 3, 4)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    batch = _fake_batch(latents=latents, response_mask=response_mask)
    extrinsic_scores = torch.zeros_like(response_mask, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="stale"):
        manager.compute_intrinsic(batch=batch, extrinsic_scores=extrinsic_scores, global_step=5)
    manager.close()


def test_maybe_train_ar_respects_explicit_warmup_gate(tmp_path):
    cfg = ResearchConfig(enabled=True)
    cfg.buffer.shard_max_samples = 64
    cfg.ar.min_buffer_samples = 1
    cfg.ar.train_every_n_steps = 2
    cfg.ar.start_after_steps = 5
    cfg.post_init()

    manager = LeaRSManager(config=cfg, checkpoint_root=str(tmp_path))
    _append_one_sample(manager, step=1)
    _append_one_sample(manager, step=3)
    _append_one_sample(manager, step=5)
    manager.buffer.flush()

    captured = {"calls": 0}

    def fake_train(sequences, global_step):
        captured["calls"] += 1
        return SimpleNamespace(train_loss=0.1, steps=1)

    manager.ar_trainer.train_from_sequences = fake_train

    metrics = manager.maybe_train_ar(global_step=4)
    assert metrics["research/ar/attempted"] == 0.0
    assert metrics["research/ar/trained"] == 0.0
    assert metrics["research/ar/skipped"] == 1.0
    assert metrics["research/ar/skip_warmup"] == 1.0
    assert captured["calls"] == 0

    metrics = manager.maybe_train_ar(global_step=5)
    manager.close()
    assert metrics["research/ar/attempted"] == 1.0
    assert metrics["research/ar/trained"] == 1.0
    assert captured["calls"] == 1


def test_maybe_train_ar_attempts_on_deterministic_cadence(tmp_path):
    cfg = ResearchConfig(enabled=True)
    cfg.buffer.shard_max_samples = 64
    cfg.ar.min_buffer_samples = 1
    cfg.ar.train_every_n_steps = 3
    cfg.ar.start_after_steps = 4
    cfg.post_init()

    manager = LeaRSManager(config=cfg, checkpoint_root=str(tmp_path))
    for step in range(1, 12):
        _append_one_sample(manager, step=step)
    manager.buffer.flush()

    captured_steps = []

    def fake_train(sequences, global_step):
        captured_steps.append(global_step)
        return SimpleNamespace(train_loss=0.1, steps=1)

    manager.ar_trainer.train_from_sequences = fake_train

    non_cadence_metrics = manager.maybe_train_ar(global_step=5)
    assert non_cadence_metrics["research/ar/skipped"] == 1.0
    assert non_cadence_metrics["research/ar/skip_cadence"] == 1.0

    for step in range(1, 12):
        manager.maybe_train_ar(global_step=step)

    manager.close()
    assert captured_steps == [4, 7, 10]
