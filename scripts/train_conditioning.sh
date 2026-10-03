#!/usr/bin/env bash
# Cluster entry point for the two text-conditioning experiments.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <unpadded|transformer> <teacher|mathwriting> [train_student.py arguments ...]" >&2
    exit 2
fi
VARIANT="$1"
CORPUS="$2"
shift 2
case "$VARIANT" in
    unpadded|transformer) ;;
    *) echo "Unknown conditioning variant: $VARIANT" >&2; exit 2 ;;
esac
case "$CORPUS" in
    teacher) CORPUS_ARGS=(--corpus data/teacher_corpus) ;;
    mathwriting) CORPUS_ARGS=(--corpus data/mathwriting_corpus --artifact none) ;;
    *) echo "Unknown corpus: $CORPUS" >&2; exit 2 ;;
esac
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec scripts/run_in_apptainer.sh "${INK_TRAIN_GPU:-0}" python scripts/train_student.py \
    --run-dir "models/student/${CORPUS}_${VARIANT}" \
    --conditioning "$VARIANT" --text-encoder-layers 2 --text-encoder-heads 4 --prefix-slots 32 \
    "${CORPUS_ARGS[@]}" "$@"
