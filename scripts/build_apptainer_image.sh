#!/usr/bin/env bash
# Build the Apptainer image: Python 3.13 + uv + a C compiler (Triton builds
# its kernel launchers at runtime), nothing else.
#
# Python packages are not baked in. scripts/run_in_apptainer.sh runs
# `uv sync` against this repo's uv.lock inside the container, so the image only
# changes when the Python or uv version does. Pulling a public image needs no
# root or Docker, which is why this does not go through a Dockerfile.
#
#   scripts/build_apptainer_image.sh            # -> .apptainer/ink_tokenizer.sif
#   INK_SIF=/path/to/ink.sif scripts/build_apptainer_image.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${INK_SIF:-$ROOT_DIR/.apptainer/ink_tokenizer.sif}"
BASE_IMAGE="${INK_BASE_IMAGE:-docker://ghcr.io/astral-sh/uv:0.9.30-python3.13-bookworm}"

# Apptainer's layer cache defaults to ~/.apptainer, which is small on clusters.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$ROOT_DIR/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$ROOT_DIR/.apptainer/tmp}"
mkdir -p "$(dirname "$SIF_PATH")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

apptainer build --force "$SIF_PATH" "$BASE_IMAGE"
echo "# Wrote $SIF_PATH"
