#!/usr/bin/env bash
# Dual-GPU pass@k code-bench evaluation.
# GPU0 (~33GB free): batch_size=4, runs smaller datasets.
# GPU1 (~48GB free): batch_size=8, runs larger datasets.
# Two waves keep both GPUs busy with balanced wall time.
set -euo pipefail

ROOT=${ROOT:-/data/ldyData/LLM-Walk-Through}
MODEL=${MODEL:-$ROOT/runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2/hf_exports/best_bench}
SANDBOX_URLS=${SANDBOX_URLS:-http://127.0.0.1:18901,http://127.0.0.1:18902,http://127.0.0.1:18903,http://127.0.0.1:18904}
OUTPUT=${OUTPUT:-$ROOT/runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2/bench_eval/best_bench_passk}
N=${N:-8}
PASS_AT=${PASS_AT:-1,4,8}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-32}
GPU0_BATCH=${GPU0_BATCH:-4}
GPU1_BATCH=${GPU1_BATCH:-8}
LOG_DIR=${LOG_DIR:-$OUTPUT/logs}

cd "$ROOT"
mkdir -p "$OUTPUT" "$LOG_DIR"

IFS=',' read -r -a SANDBOX_URL_ARRAY <<< "$SANDBOX_URLS"

run_one() {
  local gpu="$1"
  local dataset="$2"
  local batch_size="$3"
  local log_file="$LOG_DIR/${dataset}.log"

  echo "[dual-gpu] GPU${gpu} ${dataset} batch_size=${batch_size} -> ${OUTPUT}/${dataset}"
  local -a cmd=(
    uv run --extra posttrain python -m scripts.evaluate_code_bench
    --model "$MODEL"
    --bench-root data/bench
    --dataset "$dataset"
    --backend hf
    --device cuda:0
    --dtype bf16
    --attn-implementation flash_attention_2
    --batch-size "$batch_size"
    --n "$N"
    --pass-at "$PASS_AT"
    --max-tokens 512
    --temperature 0.2
    --top-p 0.95
    --timeout 10
    --max-concurrency "$MAX_CONCURRENCY"
    --prompt-style plain_dialog
    --output "$OUTPUT"
    --skip-sandbox-smoke
    --no-use-ray
  )
  for url in "${SANDBOX_URL_ARRAY[@]}"; do
    cmd+=(--sandbox-url "$url")
  done
  env CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_file" 2>&1
}

echo "[dual-gpu] wave 1: GPU0=openai_humaneval GPU1=mbppplus"
run_one 0 openai_humaneval "$GPU0_BATCH" &
pid_w1_a=$!
run_one 1 mbppplus "$GPU1_BATCH" &
pid_w1_b=$!
wait "$pid_w1_a" "$pid_w1_b"

echo "[dual-gpu] wave 2: GPU0=humanevalplus GPU1=mbpp"
run_one 0 humanevalplus "$GPU0_BATCH" &
pid_w2_a=$!
run_one 1 mbpp "$GPU1_BATCH" &
pid_w2_b=$!
wait "$pid_w2_a" "$pid_w2_b"

echo "[dual-gpu] merging summaries"
uv run python -m scripts.merge_code_bench_summaries \
  --output "$OUTPUT" \
  --pass-at "$PASS_AT"

echo "[dual-gpu] done -> ${OUTPUT}/summary.json"
