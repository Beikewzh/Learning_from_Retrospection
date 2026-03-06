#!/usr/bin/env python3
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

"""Inspect latent buffer contents."""

from __future__ import annotations

import argparse
import os

from research.buffer import ParquetLatentBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", type=str, required=True)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    buf = ParquetLatentBuffer(
        root_dir=os.path.abspath(args.buffer_dir),
        shard_max_samples=1024,
        compression="zstd",
        max_disk_gb=1000.0,
    )
    stats = buf.get_stats()
    print("Buffer stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("Sample rows:")
    for i, row in enumerate(buf.iter_rows()):
        if i >= args.show:
            break
        print(
            {
                "uid": row["uid"],
                "step": row["step"],
                "response_length": row["response_length"],
                "hidden_dim": row["hidden_dim"],
                "dtype": row["dtype"],
                "extrinsic_final": row["extrinsic_final"],
                "success": row["success"],
                "latent_blob_bytes": len(row["latent_blob"]),
            }
        )


if __name__ == "__main__":
    main()
