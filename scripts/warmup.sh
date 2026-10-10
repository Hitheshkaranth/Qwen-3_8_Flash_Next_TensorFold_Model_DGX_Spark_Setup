#!/usr/bin/env bash
# Wait for the server, then send one throwaway ~8k-token prompt so the first
# real user does not pay first-shape kernel and cache warm-up costs.
#
#   ./scripts/warmup.sh                  # uses config values below
#   BASE=http://172.17.0.1:8006 API_KEY=... ./scripts/warmup.sh
set -euo pipefail
BASE="${BASE:-http://172.17.0.1:8006}"
MODEL="${MODEL:-qwen3.8-flash-next-tf}"
API_KEY="${API_KEY:-}"
TIMEOUT_S="${TIMEOUT_S:-600}"

echo "==> waiting for $BASE/health (up to ${TIMEOUT_S}s; TensorFold loads in ~2 min)"
for ((i = 0; i < TIMEOUT_S; i += 5)); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] && break
  sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] || { echo "server not healthy" >&2; exit 1; }

echo "==> warm-up prompt (~8k tokens)"
python3 - "$BASE" "$MODEL" "$API_KEY" <<'EOF'
import json, sys, time, urllib.request, uuid
base, model, key = sys.argv[1:4]
text = f"Warm-up {uuid.uuid4()}.\n" + "".join(
    f"Entry {i:06d}: the quarterly logistics audit recorded a routine variance.\n" for i in range(400))
h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
body = json.dumps({"model": model, "max_tokens": 1,
                   "messages": [{"role": "user", "content": text + "\nSummarise in one word."}]}).encode()
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, h), timeout=600))
print(f"    {r['usage']['prompt_tokens']} tokens prefilled in {time.time() - t:.1f}s")
EOF
echo "==> ready"
