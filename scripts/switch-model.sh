#!/usr/bin/env bash
# Swap the box between TensorFold Flash-Next (:8006) and another vLLM container
# (default: Ornith on :8004). Only one large model fits in the 121 GiB at a time.
#
#   ./scripts/switch-model.sh tensorfold      # stop the other model, start TensorFold
#   ./scripts/switch-model.sh other           # stop TensorFold, start the other model
#   OTHER=vllm-qwen38-a3b ./scripts/switch-model.sh other
#
# The other container keeps `--restart unless-stopped` only while it is the live
# model, so a reboot brings back whichever was serving.
set -euo pipefail
cd "$(dirname "$0")/.."
OTHER="${OTHER:-vllm-ornith-a3b}"
KEY_FILE="${KEY_FILE:-gateway/upstream.key}"

case "${1:-}" in
  tensorfold)
    if docker ps -q -f name="^${OTHER}$" | grep -q .; then
      docker update --restart no "$OTHER" >/dev/null
      docker stop -t 60 "$OTHER"
    fi
    (cd upstream && TENSORFOLD_API_KEY="$(cat "$KEY_FILE")" ./start.sh) || true
    API_KEY="$(cat "$KEY_FILE")" ./scripts/warmup.sh
    ;;
  other)
    (cd upstream && ./stop.sh)
    docker update --restart unless-stopped "$OTHER" >/dev/null
    docker start "$OTHER"
    echo "started $OTHER; vLLM needs a few minutes before /health answers"
    ;;
  *) echo "usage: $0 tensorfold|other" >&2; exit 2 ;;
esac
