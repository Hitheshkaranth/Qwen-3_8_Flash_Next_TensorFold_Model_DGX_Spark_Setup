#!/usr/bin/env bash
# TensorFold server for azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound (Qwen3.8-Flash-Next,
# GPTQ int4 routed experts, 5 of 512 experts per token) on :8006, through MiaAI-Lab's recipe in
# ./upstream (fetched by download.sh).
#
#   ./run.sh            start (no-op if already running), then warm up
#   ./run.sh restart    stop and start with the settings below
#   ./run.sh stop       stop and remove the container
#
# Not vLLM: there is no --gpu-memory-utilization. The engine loads ~65.7 GiB of weights (including
# the FP8 n-gram table, held in memory rather than read from SSD), then sizes its sequence pool as
# MemAvailable minus 10 GiB (minus 2 GiB more for the vision helper): ~27.5 GiB here, about six
# full 262K fp8 windows. So it needs ~92 GiB free at start and cannot share the box with a 0.70
# vLLM container (Ornith on :8004 etc.) -- stop that first.
# PARALLEL 16 is the recipe's maximum (its default is 8): measured peak 400.4 tok/s at 16 users;
# at 20 the extra requests queue and aggregate drops to 321.6.
# Published only on the docker bridge (172.17.0.1), so only the gateway and Prometheus reach it.
# Bearer key on /v1/* AND /metrics, held only by the gateway (~/Documents/llm-gateway/upstream.key);
# it is passed as an environment variable and never written to upstream/.env.
# The recipe's start.sh ends with a smoke test that doesn't send the key, so it reports
# "smoke test request failed" with a key set; the warm-up below is the real check.
# Defaults to --restart unless-stopped since this is the active production model on this box.
# Only ONE large-model container should carry unless-stopped at a time -- when swapping, flip the
# other one with `docker update --restart no <container>` first. One-off test run: RESTART=no.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$SCRIPT_DIR/upstream"
NAME="qwen38-flash-next-tf"          # the recipe's CONTAINER_NAME
PORT="${PORT:-8006}"
HOST="${HOST:-172.17.0.1}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next-tf}"
PARALLEL="${PARALLEL:-16}"
VISION="${VISION:-1}"
: "${TENSORFOLD_API_KEY:=$(cat ~/Documents/llm-gateway/upstream.key 2>/dev/null || cat "$SCRIPT_DIR/gateway/upstream.key" 2>/dev/null || true)}"
export TENSORFOLD_API_KEY

[ -x "$UPSTREAM/start.sh" ] || { echo "upstream/ missing: run ./download.sh first" >&2; exit 1; }

if [ "${1:-start}" = stop ]; then
  exec "$UPSTREAM/stop.sh"
fi

cat > "$UPSTREAM/.env" <<EOF
# Written by run.sh -- edit run.sh, not this file. No secrets here.
PORT=$PORT
HOST=$HOST
SERVED_NAME=$SERVED_NAME
PARALLEL=$PARALLEL
VISION=$VISION
EOF

MODE=(); [ "${1:-start}" = restart ] && MODE=(restart)
"$UPSTREAM/start.sh" "${MODE[@]}" || echo "(start.sh's key-less smoke test fails with a key set; checking with a real request)"
docker update --restart "${RESTART:-unless-stopped}" "$NAME" >/dev/null

# Warm-up: wait for health, then prefill one ~8K-token prompt so the first user
# doesn't pay first-shape kernel costs.
for _ in $(seq 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health")" = 200 ] && break
  sleep 5
done
python3 - "http://$HOST:$PORT" "$SERVED_NAME" "$TENSORFOLD_API_KEY" <<'PY'
import json, sys, time, urllib.request, uuid
base, model, key = sys.argv[1:4]
text = f"Warm-up {uuid.uuid4()}.\n" + "".join(
    f"Entry {i:06d}: the quarterly logistics audit recorded a routine variance.\n" for i in range(400))
h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
body = json.dumps({"model": model, "max_tokens": 1,
                   "messages": [{"role": "user", "content": text + "\nSummarise in one word."}]}).encode()
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, h), timeout=600))
print(f"ready: {model} on {base} ({r['usage']['prompt_tokens']}-token warm-up in {time.time() - t:.1f}s)")
PY
