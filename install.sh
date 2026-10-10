#!/usr/bin/env bash
# Installer for Qwen3.8-Flash-Next (INT4-AutoRound, top-5 experts) served by TensorFold's
# Zig engine on a single DGX Spark (GB10), 16 concurrent users.
#
#   git clone https://github.com/Hitheshkaranth/Qwen-3_8_Flash_Next_TensorFold_Model_DGX_Spark_Setup.git
#   cd Qwen-3_8_Flash_Next_TensorFold_Model_DGX_Spark_Setup && ./install.sh
#
# Clones MiaAI-Lab's TensorFold recipe (MIT) at the pinned commit into ./upstream,
# applies patches/, copies config/tensorfold.env to upstream/.env, pulls the
# prebuilt engine image (~25 GiB) and the checkpoint (~122 GiB), starts the
# server behind the gateway's upstream key, and warms it up. Safe to re-run.
#
# Env overrides: UPSTREAM_REF, API_KEY (defaults to gateway/upstream.key),
# SKIP_START=1 (download only).
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
UPSTREAM_URL="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git"
UPSTREAM_REF="${UPSTREAM_REF:-364f21a1a46a366fa39c4da0d53b7b1f38624610}"

echo "==> Checking prerequisites"
for c in git docker curl python3; do
  command -v "$c" >/dev/null || { echo "$c is required" >&2; exit 1; }
done
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable" >&2; exit 1; }
free_gib=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gib" -ge 160 ] || echo "WARNING: ${free_gib} GiB free; checkpoint + image need ~150 GiB." >&2

echo "==> TensorFold recipe @ ${UPSTREAM_REF:0:7}"
[ -d upstream/.git ] || git clone -q "$UPSTREAM_URL" upstream
git -C upstream fetch -q origin
git -C upstream checkout -q "$UPSTREAM_REF"
for p in patches/*.patch; do
  if git -C upstream apply --check "$HERE/$p" 2>/dev/null; then
    git -C upstream apply "$HERE/$p" && echo "    applied $p"
  else
    echo "    $p already applied (or not applicable), skipping"
  fi
done
cp config/tensorfold.env upstream/.env

key="${API_KEY:-}"
[ -z "$key" ] && [ -f gateway/upstream.key ] && key="$(cat gateway/upstream.key)"
[ -n "$key" ] || echo "    no API key: the server will be open on 172.17.0.1:8006. Run gateway/setup.sh first to lock it."

echo "==> Engine image + checkpoint (Xet disabled; it stalls on some networks)"
(cd upstream && MEM_CHECK=0 HF_HUB_DISABLE_XET=1 ./scripts/prepare.sh)

[ "${SKIP_START:-0}" = 1 ] && { echo "==> SKIP_START=1, done"; exit 0; }

echo "==> Starting (needs ~92 GiB free: stop any other large model first)"
# start.sh's built-in smoke test does not send the key, so with a key it reports
# a 401 "failure" while the server is fine; warmup.sh below is the real check.
(cd upstream && TENSORFOLD_API_KEY="$key" ./start.sh) || true
API_KEY="$key" ./scripts/warmup.sh
echo "==> Serving qwen3.8-flash-next-tf on http://172.17.0.1:8006 (through the gateway: :8080)"
