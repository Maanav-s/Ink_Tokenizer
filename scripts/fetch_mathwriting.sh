#!/usr/bin/env bash
# Download MathWriting into data/mathwriting-2024 and convert its human inks
# into student corpus shards in data/mathwriting_corpus (no W&B push).
#
#   scripts/fetch_mathwriting.sh [mathwriting_corpus.py args ...]
#
# Rerunning skips the steps that are already done. The raw dataset needs about
# 3 GB to download and 9 GB extracted; the archive is deleted after extracting.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
DATA="$ROOT_DIR/data"
mkdir -p "$DATA"

if [[ ! -f "$DATA/mathwriting-2024/readme.md" ]]; then
    if [[ ! -f "$DATA/mathwriting-2024.tgz" ]]; then
        echo "downloading $URL"
        curl -fL --retry 3 -o "$DATA/mathwriting-2024.tgz.part" "$URL"
        mv "$DATA/mathwriting-2024.tgz.part" "$DATA/mathwriting-2024.tgz"
    fi
    echo "extracting"
    tar xzf "$DATA/mathwriting-2024.tgz" -C "$DATA"
    rm "$DATA/mathwriting-2024.tgz"
fi

if [[ -f "$DATA/mathwriting_corpus/meta.json" ]]; then
    echo "data/mathwriting_corpus already converted"
else
    # A partial conversion has shards but no meta.json; start it over.
    rm -rf "$DATA/mathwriting_corpus"
    cd "$ROOT_DIR"
    uv run --all-extras python scripts/mathwriting_corpus.py "$@"
fi
