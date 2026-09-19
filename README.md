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

## Online-ink representation

`ink_tokenizer.scribe` adds a ScribeTokens-style codec for native pen data:
eight unit directions plus `DOWN`/`UP`, with Bresenham decomposition and an
optional deterministic BPE layer. It is a training representation for a future
structural predictor, not a replacement for the InkSight image-model codec.
Its base vocabulary is always 10 tokens and BPE never crosses a pen-state
boundary, which keeps partial-stroke handling explicit.

```python
from ink_tokenizer import Ink, ScribeBPE, encode_scribe

encoding = encode_scribe(Ink.from_point_lists([[(10, 10), (22, 10)]]), step=2)
model = ScribeBPE.fit([encoding.tokens], vocabulary_size=32)
assert model.decode_ids(model.encode_ids(encoding.tokens)) == encoding.tokens
```

The encoding retains `origin` and `step` as example metadata because the fixed
direction vocabulary intentionally does not encode an absolute page position.

To inspect recovered ink, render either raw `Ink` JSON or the output from
`derender -o` as a self-contained SVG.  When the original image is still
available, it is embedded behind the blue recovered strokes.

```bash
uv run ink-tokenizer visualize ink.json -o ink.svg --show-order
```


## Edge + OCR baseline

The optional `experiments` extra contains a conventional full-page baseline:
Canny contours propose closed flowchart regions and RapidOCR reads sparse text.
It is evaluation tooling, not the product suggestion path.

```bash
uv sync --extra experiments
uv run python scripts/benchmark_edge_ocr.py \
  artifacts/datasets/FC_database_offline_1.0/FC_1.0_offline_extension \
  artifacts/comparisons/fc_offline_edge_ocr
```

The default contour gate is deliberately conservative (`0.55`). Pass
`--min-contour-score 0` to inspect the unfiltered detector.

## Docs

- [docs/inksight.md](docs/inksight.md) — why InkSight, how it's integrated,
  and the constraints it imposes
- [docs/running-inference.md](docs/running-inference.md) — backends, GPU,
  air-gapped and remote setups

Portions derived from [InkSight](https://github.com/google-research/inksight)
(Apache-2.0) — see [NOTICE](NOTICE).
