#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/ldyData/LLM-Walk-Through}
CKPT=${CKPT:-$ROOT/runs/walkie_code_0.5b/latest.pt}
TOKENIZER=${TOKENIZER:-$ROOT/data/cache/walkie_code/tokenizer.json}
MODEL=${MODEL:-$ROOT/runs/walkie_code_0.5b_vllm_hf}
SANDBOX_URL=${SANDBOX_URL:-http://127.0.0.1:18901}
DATASET=${DATASET:-all}
LIMIT=${LIMIT-}
N=${N:-8}
PASS_AT=${PASS_AT:-1,4,8}
OUTPUT=${OUTPUT:-$ROOT/runs/eval/walkie_code_0.5b_passk}
BACKEND=${BACKEND:-hf}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-2}

cd "$ROOT"

uv run --extra walkie python -m scripts.export_walkie_to_hf \
  --checkpoint "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --output "$MODEL"

eval_args=(
  uv run --extra posttrain python -m scripts.evaluate_code_bench
  --config configs/eval/walkie_code_bench_passk.yaml
  --model "$MODEL"
  --sandbox-url "$SANDBOX_URL"
  --dataset "$DATASET"
  --backend "$BACKEND"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --n "$N"
  --pass-at "$PASS_AT"
  --output "$OUTPUT"
  --no-use-ray
)
if [[ -n "$LIMIT" ]]; then
  eval_args+=(--limit "$LIMIT")
fi
"${eval_args[@]}"
