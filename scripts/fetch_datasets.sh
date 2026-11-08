#!/usr/bin/env bash
# Download the human handwriting datasets, convert each into corpus shards, and
# pool them with MathWriting into data/pooled_corpus (no W&B push).
#
#   scripts/fetch_datasets.sh --accept-licenses [iamondo] [deepwriting] [brush] [mathwriting]
#
# With no dataset named, all four. Rerunning skips the steps that are already
# done. The pool is built once, from whichever corpora exist then; remove
# data/pooled_corpus to pool again after adding a dataset.
#
# Disk:
#   iamondo      31 MB download, 123 MB extracted (the archive is deleted)
#   deepwriting  135 MB zip, read in place
#   brush        567 MB zip, read in place
#   mathwriting  3 GB download, 9 GB extracted (scripts/fetch_mathwriting.sh)
# plus about 100-200 MB per converted text corpus.
#
# DeepWriting and BRUSH are only downloaded with --accept-licenses:
#   DeepWriting  CC BY-NC-SA 4.0 and no distribution: keep this one copy, and do
#                not push its corpus, or a pool containing it, to W&B or copy it
#                to another machine. Its authors also ask users to register for
#                IAM-OnDB. https://files.ait.ethz.ch/projects/deepwriting/license.pdf
#   BRUSH        non-commercial research only; cite Kotani et al., ECCV 2020.
#                https://github.com/brownvc/decoupled-style-descriptors
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT_DIR/data"
IAMONDO_URL="http://www.iapr-tc11.org/dataset/IAMonDo/IAMonDo-db-1.0.tar.gz"
DEEPWRITING_URL="https://files.ait.ethz.ch/projects/deepwriting/extended_dataset.zip"
BRUSH_URL="https://drive.usercontent.google.com/download?id=1NIIXDfmpUhI6i80Dg2363PIdllY7FRVQ&export=download&confirm=t"

accept=0
datasets=()
for arg in "$@"; do
    case "$arg" in
        --accept-licenses) accept=1 ;;
        iamondo|deepwriting|brush|mathwriting) datasets+=("$arg") ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[[ ${#datasets[@]} -gt 0 ]] || datasets=(iamondo deepwriting brush mathwriting)

wants() {
    [[ " ${datasets[*]} " == *" $1 "* ]]
}

if (( ! accept )) && { wants deepwriting || wants brush; }; then
    sed -n '/^# DeepWriting and BRUSH/,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
    echo "Rerun with --accept-licenses to download them, or name only the other datasets." >&2
    exit 1
fi

download() {
    local url="$1" dest="$2"
    [[ -f "$dest" ]] && return
    mkdir -p "$(dirname "$dest")"
    echo "downloading $url"
    curl -fL --retry 3 -o "$dest.part" "$url"
    # Google Drive answers with an HTML page instead of the file when its quota is exceeded.
    if [[ "$dest" == *.zip && "$(head -c 2 "$dest.part")" != "PK" ]]; then
        echo "$url did not return a zip file; try again later" >&2
        rm "$dest.part"
        exit 1
    fi
    mv "$dest.part" "$dest"
}

convert() {
    local out="$DATA/$1_corpus"
    shift
    if [[ -f "$out/meta.json" ]]; then
        echo "data/$(basename "$out") already converted"
        return
    fi
    # A partial conversion has shards but no meta.json; start it over.
    rm -rf "$out"
    (cd "$ROOT_DIR" && uv run --all-extras python "$@")
}

mkdir -p "$DATA"
if wants iamondo; then
    if [[ ! -f "$DATA/IAMonDo-db-1.0/0.set" ]]; then
        download "$IAMONDO_URL" "$DATA/IAMonDo-db-1.0.tar.gz"
        tar xzf "$DATA/IAMonDo-db-1.0.tar.gz" -C "$DATA"
        rm "$DATA/IAMonDo-db-1.0.tar.gz"
    fi
    convert iamondo scripts/iamondo_corpus.py
fi
if wants deepwriting; then
    download "$DEEPWRITING_URL" "$DATA/deepwriting/extended_dataset.zip"
    convert deepwriting scripts/deepwriting_corpus.py
fi
if wants brush; then
    download "$BRUSH_URL" "$DATA/brush/refined_BRUSH.zip"
    convert brush scripts/brush_corpus.py
fi
if wants mathwriting; then
    "$ROOT_DIR/scripts/fetch_mathwriting.sh"
fi

text=()
for name in iamondo deepwriting brush; do
    [[ -f "$DATA/${name}_corpus/meta.json" ]] && text+=("data/${name}_corpus")
done
if [[ -f "$DATA/pooled_corpus/meta.json" ]]; then
    echo "data/pooled_corpus already pooled; remove it to pool again"
elif [[ -f "$DATA/mathwriting_corpus/meta.json" && ${#text[@]} -gt 0 ]]; then
    rm -rf "$DATA/pooled_corpus"
    (cd "$ROOT_DIR" && uv run --all-extras python scripts/pool_corpus.py --text "${text[@]}")
else
    echo "not pooling: needs data/mathwriting_corpus and at least one text corpus"
fi
