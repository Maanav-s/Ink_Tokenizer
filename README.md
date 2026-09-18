# Ink Tokenizer

Structural autocomplete for handwritten engineering work — recognizing math,
truth tables and circuit diagrams as they're written on a whiteboard, and
completing the *structure* of what comes next.

See [AGENTS.md](AGENTS.md) for the project objective and scope.

## Setup

**Linux (to run inference):**

```bash
./scripts/setup_linux.sh --prefetch
uv run ink-tokenizer derender word.jpg
```

Installs `uv` if missing (no root), fetches a suitable Python, installs
TensorFlow, downloads the checkpoint, and reports readiness.

**Any platform (codec, geometry, tests):**

```bash
uv sync --extra dev
uv run pytest
```

TensorFlow is not required and is not installed on Windows — see
[docs/running-inference.md](docs/running-inference.md) for driving a remote
Linux box instead.

## Use

```python
from PIL import Image
from ink_tokenizer import get_backend     # in-process if TF is here, else HTTP

with get_backend() as backend:
    result = backend.derender_image(Image.open("word.jpg"))
    print(result.text, result.ink.num_points)
```

`ink-tokenizer doctor` reports what the current machine can do.

## Docs

- [docs/inksight.md](docs/inksight.md) — why InkSight, how it's integrated,
  and the constraints it imposes
- [docs/running-inference.md](docs/running-inference.md) — backends, GPU,
  air-gapped and remote setups

Portions derived from [InkSight](https://github.com/google-research/inksight)
(Apache-2.0) — see [NOTICE](NOTICE).
