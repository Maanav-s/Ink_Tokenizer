#!/usr/bin/env bash
# Bootstrap a Linux machine for InkSight inference.
#
#   ./scripts/setup_linux.sh              # install deps, report readiness
#   ./scripts/setup_linux.sh --prefetch   # also download the 692 MB checkpoint
#   ./scripts/setup_linux.sh --no-uv      # force venv + pip instead of uv
#
# Safe to re-run. Prefers uv, installing it into ~/.local/bin if absent: it
# needs no root and fetches its own Python, which matters on a lab box where
# the system Python is too old and `python3-venv` may not be installed.
set -euo pipefail

cd "$(dirname "$0")/.."

PREFETCH=0
USE_UV=1
for arg in "$@"; do
    case "$arg" in
        --prefetch) PREFETCH=1 ;;
        --no-uv) USE_UV=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

case "$(uname -s)" in
    Linux|Darwin) ;;
    *)
        echo "This script targets Linux/macOS: tensorflow-text has no Windows wheels." >&2
        exit 1
        ;;
esac

echo "==> Environment"
echo "    $(uname -s -m)"

if [ "$USE_UV" = "1" ] && ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found; installing to ~/.local/bin (no root required)"
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "    uv install failed; falling back to venv + pip" >&2
        USE_UV=0
    fi
fi

if [ "$USE_UV" = "1" ] && command -v uv >/dev/null 2>&1; then
    echo "==> Installing with uv ($(uv --version))"
    # uv reads requires-python and .python-version and will fetch a matching
    # interpreter itself, so the system Python version does not matter.
    uv sync --extra inference --extra service --extra dev
    RUN=(uv run)
else
    echo "==> Installing with venv + pip"
    python3 - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    sys.exit(
        f"Need Python >=3.11,<3.14 (InkSight's range); found "
        f"{sys.version.split()[0]}. Install a supported Python, or drop "
        f"--no-uv and let uv fetch one."
    )
PY
    if [ ! -d .venv ]; then
        if ! python3 -m venv .venv 2>/dev/null; then
            echo "" >&2
            echo "Could not create a virtualenv. Most likely the venv module is" >&2
            echo "missing. Either:" >&2
            echo "    sudo apt install python3-venv     # needs root" >&2
            echo "    ./scripts/setup_linux.sh          # or let uv handle it" >&2
            exit 1
        fi
    fi
    # shellcheck disable=SC1091
    . .venv/bin/activate
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -e ".[inference,service,dev]"
    RUN=()
fi

echo "==> Tests (no model download needed)"
"${RUN[@]}" pytest -q

if [ "$PREFETCH" = "1" ]; then
    echo "==> Prefetching checkpoint (~692 MB)"
    "${RUN[@]}" python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Derendering/InkSight-Small-p'))
"
fi

echo "==> Readiness"
"${RUN[@]}" ink-tokenizer doctor

cat <<'EOF'

Ready. Next:
  ink-tokenizer derender path/to/word.jpg     # in-process, no server needed
  uvicorn app:app --app-dir services/inksight --host 0.0.0.0 --port 8000
                                              # serve other machines
EOF
