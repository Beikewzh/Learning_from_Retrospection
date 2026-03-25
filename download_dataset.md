# Download Model and Datasets

This guide downloads:
- `Qwen/Qwen3-4B` to `models/Qwen__Qwen3-4B`
- `hiyouga/math12k` to `data/math12k/train.parquet`
- `HuggingFaceH4/MATH-500` to `data/math500/test.parquet`

## 1) Enter the repo and open the container

```bash
cd ~/links/scratch/Learning_from_Retrospection

apptainer shell --nv \
  --bind "${PWD}:/workspace" \
  easyr1.sif
```

## 2) Inside container: move to workspace

```bash
cd /workspace
```

## 3) Download model

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir models/Qwen__Qwen3-4B \
  --exclude "*.gguf"
```

## 4) Download datasets and save as parquet

```bash
python3 - <<'EOF'
from datasets import load_dataset
import os

os.makedirs("data/math12k", exist_ok=True)
os.makedirs("data/math500", exist_ok=True)

ds = load_dataset("hiyouga/math12k", split="train")
ds.to_parquet("data/math12k/train.parquet")
print(f"math12k: {len(ds)} rows saved")

ds2 = load_dataset("HuggingFaceH4/MATH-500", split="test")
ds2.to_parquet("data/math500/test.parquet")
print(f"MATH-500: {len(ds2)} rows saved")
EOF
```