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

from research.buffer.parquet_buffer import ParquetLatentBuffer


def test_parquet_buffer_append_and_read(tmp_path):
    buffer = ParquetLatentBuffer(
        root_dir=str(tmp_path),
        shard_max_samples=2,
        compression="zstd",
        max_disk_gb=1.0,
    )

    latents = torch.randn(3, 5, 4)
    response_lengths = torch.tensor([5, 4, 1])
    extrinsic = torch.tensor([1.0, 0.0, -1.0])
    success = torch.tensor([True, False, False])

    buffer.append_batch(
        uids=["a", "b", "c"],
        step=3,
        latents=latents,
        response_lengths=response_lengths,
        extrinsic_final=extrinsic,
        success=success,
        dtype="fp16",
    )
    buffer.flush()

    stats = buffer.get_stats()
    assert stats["buffer/total_samples"] == 2
    assert stats["buffer/total_shards"] >= 1

    seqs = buffer.load_sequences(max_samples=8, seed=1)
    assert len(seqs) == 2
    assert all(s.ndim == 2 for s in seqs)
    assert sorted([s.shape[0] for s in seqs]) == [4, 5]
