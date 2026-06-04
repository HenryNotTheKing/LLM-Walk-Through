#!/usr/bin/env bash
set -euo pipefail

cd /data/ldyData/LLM-Walk-Through

uv run --extra walkie --extra flash --extra posttrain python -m scripts.run_sft_bench_loop \
  --config configs/train/sft_walkie_kodcode_bench.yaml \
  --init-from runs/walkie_code_0.5b/latest.pt \
  --out-dir runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2 \
  --gpu 0,1 \
  --eval-gpu 1 \
  --segment-steps 1000 \
  --total-steps 10000 \
  --full-eval-interval 1000 \
  --skip-sandbox-smoke