#!/usr/bin/env bash
# Fetch everything TensorFold needs: MiaAI-Lab's recipe (pinned), its prebuilt
# Zig-engine image (~25 GiB, pulled from GHCR; built locally only if no matching
# tag exists) and azampatti's INT4-AutoRound checkpoint (~122 GiB on disk, into
# the Hugging Face cache ~/.cache/huggingface, not ./models).
# Plain HTTP: Xet transfers stalled on this box, and the recipe's in-container
# downloader doesn't pass HF_HUB_DISABLE_XET through, so we patch that one line.
# MEM_CHECK=0 skips the recipe's serving-memory preflight, which would refuse to
# download while another model is resident. Safe to re-run: every step resumes.
set -euo pipefail
cd "$(dirname "$0")"

RECIPE_URL="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git"
RECIPE_REF="${RECIPE_REF:-364f21a1a46a366fa39c4da0d53b7b1f38624610}"

[ -e upstream ] || git clone -q "$RECIPE_URL" upstream
git -C upstream fetch -q origin
git -C upstream -c advice.detachedHead=false checkout -q "$RECIPE_REF" 2>/dev/null ||
  echo "upstream has local changes; leaving it at $(git -C upstream rev-parse --short HEAD)"

if ! grep -q 'HF_HUB_DISABLE_XET' upstream/scripts/prepare.sh; then
  sed -i 's#      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \\#      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-}" \\#' \
    upstream/scripts/prepare.sh
fi

cd upstream
exec env MEM_CHECK=0 HF_HUB_DISABLE_XET=1 ./scripts/prepare.sh
