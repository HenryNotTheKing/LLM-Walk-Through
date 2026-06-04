#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/ldyData/LLM-Walk-Through}
CKPT=${CKPT:-$ROOT/runs/walkie_code_0.5b/latest.pt}
TOKENIZER=${TOKENIZER:-$ROOT/data/cache/walkie_code/tokenizer.json}
MODEL=${MODEL:-$ROOT/runs/walkie_code_0.5b_vllm_hf}
SANDBOX_URL=${SANDBOX_URL:-http://127.0.0.1:18901}
DATASET=${DATASET:-openai_humaneval}
LIMIT=${LIMIT-2}
N=${N:-1}
PASS_AT=${PASS_AT:-1}
OUTPUT=${OUTPUT:-$ROOT/runs/eval/walkie_code_0.5b}

cd "$ROOT"

uv run --extra walkie python -m scripts.export_walkie_to_hf \
  --checkpoint "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --output "$MODEL"

eval_args=(
  uv run --extra posttrain python -m scripts.evaluate_code_bench
  --config configs/eval/walkie_code_bench.yaml
  --model "$MODEL"
  --sandbox-url "$SANDBOX_URL"
  --dataset "$DATASET"
  --n "$N"
  --pass-at "$PASS_AT"
  --output "$OUTPUT"
  --no-use-ray
)
if [[ -n "$LIMIT" ]]; then
  eval_args+=(--limit "$LIMIT")
fi
"${eval_args[@]}"