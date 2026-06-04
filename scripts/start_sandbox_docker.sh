#!/usr/bin/env bash
# Start MultiModal-Jupyter-Sandbox with 4 host ports (18901-18904).
set -euo pipefail

IMAGE=${IMAGE:-crpi-jradxyzaujpkblrm.cn-hangzhou.personal.cr.aliyuncs.com/henrynottheking/multimodal-ipython-sandbox:latest}
NAME=${NAME:-walkie-code-sandbox-eval}

docker stop "$NAME" 2>/dev/null || true
docker rm "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  -p 18901:18901 -p 18902:18902 -p 18903:18903 -p 18904:18904 \
  --restart unless-stopped \
  "$IMAGE" \
  bash ./start_serving.sh

echo "waiting for sandbox ports..."
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:18901/docs >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

for port in 18901 18902 18903 18904; do
  curl -sf -X POST "http://127.0.0.1:${port}/jupyter_sandbox" \
    -H 'Content-Type: application/json' \
    -d '{"session_id":"smoke","code":"print(\"ALL TESTS PASSED\")","timeout":10}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='success'; print(f'port {port}: OK')"
done

echo "sandbox ready on ports 18901-18904"
