#!/usr/bin/env bash
# One-line installer for Qwen3.8-Flash-Next (INT4-AutoRound, TensorFold) on a DGX Spark (GB10).
#
#   curl -fsSL https://raw.githubusercontent.com/Hitheshkaranth/Qwen-3_8_Flash_Next_TensorFold_Model_DGX_Spark_Setup/main/install.sh | bash
#
# Clones this repo (or updates it if already present in the current directory), fetches the
# recipe, engine image and checkpoint, and starts the server via run.sh. Every step skips work
# that's already done, so it's safe to re-run.
set -euo pipefail

REPO_URL="${TF_REPO_URL:-https://github.com/Hitheshkaranth/Qwen-3_8_Flash_Next_TensorFold_Model_DGX_Spark_Setup.git}"
DIR="Qwen-3_8_Flash_Next_TensorFold_Model_DGX_Spark_Setup"

echo "==> Checking prerequisites..."
for c in git docker curl python3; do
  command -v "$c" >/dev/null 2>&1 || { echo "$c is required. Install it first." >&2; exit 1; }
done
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon isn't reachable (is it running, and do you have permission to use it?)." >&2
  exit 1
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "Docker lists no NVIDIA runtime. Install the NVIDIA Container Toolkit, then re-run:" >&2
  echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  exit 1
fi
free_gib=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gib" -ge 160 ] || echo "WARNING: ${free_gib} GiB free in \$HOME; the image + checkpoint need ~150 GiB." >&2

if [ -f run.sh ] && [ -f download.sh ] && grep -q TensorFold run.sh 2>/dev/null; then
  echo "==> Using the repo in $(pwd)"
elif [ -d "$DIR/.git" ]; then
  echo "==> Updating $DIR..."
  git -C "$DIR" pull --ff-only
  cd "$DIR"
else
  echo "==> Cloning $DIR..."
  git clone "$REPO_URL" "$DIR"
  cd "$DIR"
fi

echo "==> Fetching recipe, engine image (~25 GiB) and checkpoint (~122 GiB)..."
./download.sh

echo "==> Starting the server (stop any other large model first; this one needs ~92 GiB free)..."
./run.sh

cat <<'EOF'

Done. Qwen3.8-Flash-Next is serving as qwen3.8-flash-next-tf on http://172.17.0.1:8006/v1.
Put the metering gateway in front for per-user keys and usage: ./gateway/setup.sh (see docs/GATEWAY.md).
EOF
