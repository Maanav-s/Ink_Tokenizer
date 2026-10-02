#!/usr/bin/env bash
# Run a command inside the project's Apptainer image, from the repo root,
# with the repo's .venv synced to uv.lock.
#
#   scripts/run_in_apptainer.sh cpu python scripts/text_to_ink.py "Hello world"
#   scripts/run_in_apptainer.sh 0 python scripts/train_student.py ...
#   scripts/run_in_apptainer.sh 0,1 bash
#
# The first argument is CUDA_VISIBLE_DEVICES, or "cpu" to run without a GPU.
# Extras to sync come from INK_EXTRAS (default: "model render").
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${INK_SIF:-$ROOT_DIR/.apptainer/ink_tokenizer.sif}"
EXTRAS="${INK_EXTRAS:-model render}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <cuda_visible_devices|cpu> <command> [args ...]" >&2
    exit 2
fi
DEVICES="$1"
shift

if [[ ! -f "$SIF_PATH" ]]; then
    echo "Apptainer image not found: $SIF_PATH (run scripts/build_apptainer_image.sh)" >&2
    exit 1
fi

APPTAINER_ARGS=(--bind "$ROOT_DIR:$ROOT_DIR" --pwd "$ROOT_DIR")
if [[ "$DEVICES" == "cpu" ]]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES=""
else
    APPTAINER_ARGS+=(--nv)
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$DEVICES"
fi

# Keep uv's multi-GB wheel cache and Triton's compiled kernels (mamba-ssm)
# next to the repo, not in a small-quota $HOME.
export APPTAINERENV_UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export APPTAINERENV_TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT_DIR/.cache/triton}"
# Likewise W&B's run files and artifact staging/cache.
export APPTAINERENV_WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/.cache/wandb}"
export APPTAINERENV_WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT_DIR/.cache/wandb/cache}"
export APPTAINERENV_WANDB_DATA_DIR="${WANDB_DATA_DIR:-$ROOT_DIR/.cache/wandb/data}"
mkdir -p "$APPTAINERENV_WANDB_DIR"
# The image's own Python, so .venv never points at a host interpreter.
export APPTAINERENV_UV_PYTHON_DOWNLOADS=never

SYNC_ARGS=()
for extra in $EXTRAS; do
    SYNC_ARGS+=(--extra "$extra")
done

# INK_NO_SYNC=1 skips the sync. Use it for jobs that start together on a
# cluster, so they don't race over the same .venv (sync once beforehand).
if [[ "${INK_NO_SYNC:-0}" != "1" ]]; then
    apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_PATH" uv sync --quiet "${SYNC_ARGS[@]}"
fi
exec apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_PATH" uv run --no-sync "$@"
