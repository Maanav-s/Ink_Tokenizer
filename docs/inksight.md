# InkSight integration

How this project uses [InkSight](https://github.com/google-research/inksight)
to bootstrap offline (scanned/photographed) handwriting into online-style ink,
and what that ink actually is once you have it.

## What InkSight gives us

A ViT + mT5 encoder-decoder that reads an image of handwriting and emits a
token sequence encoding **stroke order and point order** — the "online" signal
a scan doesn't carry. Published in TMLR 2025.

| | |
|---|---|
| Code | `google-research/inksight`, Apache-2.0 |
| Weights | `Derendering/InkSight-Small-p`, Apache-2.0, **692 MB** |
| Format | TensorFlow SavedModel |
| Runtime | TF 2.20.0 + tensorflow-text 2.20.1 |
| Python | `>=3.11,<3.14` |
| Input | one **224×224** padded crop |
| Prompts | `Derender the ink.` · `Recognize and derender.` · `Derender the ink: <text>` |

Only **Small-p** is public. There is no PyTorch or ONNX port, so TensorFlow is
non-negotiable for the forward pass.

## How it is wired up here

```
ink_tokenizer/            no TensorFlow, runs natively on Windows
  ink.py                  Stroke / Ink data model + JSON
  codec.py                ink-token <-> Ink  (the project's actual subject)
  preprocess.py           crop -> 224x224 padding, and its exact inverse
  client.py               HTTP client for the service
services/inksight/        Linux container, the only thing that needs TF
  app.py                  FastAPI: padded crops in, raw model text out
  tf_compat.py            XLA flags the checkpoint depends on
```

The service boundary is **"image in → raw token text out."** Nothing else
crosses it. Geometry and the token codec stay in `ink_tokenizer/`, where they
are dependency-free and unit-testable without a GPU, a container, or a 692 MB
download.

### Why this split, and not the alternatives

**Why not vendor the InkSight repo wholesale?** There is barely a library to
vendor. Upstream is a notebook plus ~9 KB of `utils/`, not a package, and it
is not on PyPI. The part we actually need — `text_to_tokens` / `detokenize` —
is *inside `colab.ipynb`*, is ~40 lines, and imports nothing. Vendoring the
repo would drag TF 2.20, docTR and Jupyter into a project that will later be
torch-based, to obtain code that fits on one screen. So: the codec is ported
into `codec.py` (Apache-2.0, attributed in [NOTICE](../NOTICE)), and the model
is containerized.

**Why a service rather than an in-process import?** `tensorflow-text` publishes
**no Windows wheels** — Linux and macOS-arm64 only. On this machine InkSight
cannot run in-process at all. Even on Linux, pinning TF 2.20 in the main
project would collide with the training stack later. A service makes the
dependency someone else's problem permanently, and makes it trivial to swap
InkSight out if a better derenderer appears.

**Why not the hosted HF Space?** Fine for eyeballing one image. Not for
bootstrapping a dataset: rate-limited, and it means uploading your notes to a
third party.

## Running it

Docker (reproducible; matches upstream's pins):

```bash
cd services/inksight
docker compose up --build        # first run downloads 692 MB into a volume
```

WSL2 directly (faster to a first result, no daemon):

```bash
wsl -d Ubuntu
python3 -m venv .venv && source .venv/bin/activate
pip install -r services/inksight/requirements.txt
cd services/inksight && uvicorn app:app --host 0.0.0.0 --port 8000
```

Then, from Windows:

```python
from PIL import Image
from ink_tokenizer import InkSightClient

with InkSightClient() as client:
    result = client.derender_image(Image.open("word.jpg"))
    print(result.text, result.ink.num_points)
```

GPU is optional. The RTX 4050's 6 GB fits Small-p comfortably; switch
`requirements.txt` to `tensorflow[and-cuda]==2.20.0` and uncomment the `deploy`
block in `compose.yaml`.

## Constraints that shape the rest of the project

These are properties of InkSight, not of our wrapper. Design around them.

**1. There are no timestamps.** InkSight recovers *order*, not *time*. An
`Ink` is ordered points with no `t`, so there is no velocity, no pressure, no
pen-up duration. Anything downstream that wants temporal features will have to
synthesize timing, and that synthetic timing will not match real tablet ink.
This is the single biggest caveat for calling the output "online ink."

**2. Coordinates are quantized to a 225×225 grid.** Tokens address integers
`0..224` in *padded-crop* space. Within one word-sized crop that is fine; it is
not fine if you feed it a large region, since the whole region collapses onto
225 levels per axis. Crop tightly.

**3. The model only ever sees one 224×224 crop.** Full pages require detecting
regions, running each, then mapping every ink back to page coordinates —
`crop_and_pad` returns the transform that does this.

**4. Region detection is where our use case diverges from upstream — and this
is the real risk.** Upstream's page pipeline gets its boxes from an OCR *word*
detector (Google Cloud Vision in the paper; docTR or Tesseract as free
fallbacks). Those detectors are trained to find **words of text**. Our inputs
are truth tables, circuit diagrams and equations. A word detector will not
sensibly box a K-map, a NAND gate, or a fraction bar, and the paper's own
numbers come from the Cloud Vision detector, not the free ones — so quality
degrades twice over on our domain.

Expect to replace the detection stage rather than adopt it: connected-component
or stroke-cluster segmentation, or a layout model trained on diagrams. Treat
the derenderer and the segmenter as independent problems. `derender_boxes`
takes boxes from *any* source for exactly this reason.

**5. Empty outputs are normal.** `Recognize and derender.` sometimes returns no
ink; upstream retries with `Derender the ink.`. The service does this
automatically and reports it as `used_fallback`.

**6. Do not bump the TensorFlow pins casually.** The checkpoint predates TF
2.18's XLA GEMM lowering change. `tf_compat.py` restores the old behavior via
`XLA_FLAGS`, set *before* TF is first imported — which is why the service
imports `tf` from that module and never `import tensorflow` directly. The model
card's older advice to pin ≤2.17 is the alternative route to the same place.

## Verified end to end

A fresh Linux checkout was bootstrapped and run against the real checkpoint
(CPU-only WSL2, Ubuntu 24.04). `scripts/setup_linux.sh --prefetch` installed
uv, fetched CPython 3.13.15 (the system Python was 3.12), installed TF 2.20.0
and tensorflow-text 2.20.1, and downloaded the checkpoint. On upstream's
`test_inputs/word.jpg` (250x126):

```
recognized text : 'Neat:'
strokes/points  : 11 / 145
bbox src coords : 1.1,10.0 .. 248.9,98.2   (inside the source image)
model load      : 11.5 s
inference       : 180.7 s  (CPU, one word)
```

The bounding box landing inside the source image is the check that matters:
it means the 224x224 pad transform was inverted correctly and the ink is in
image coordinates, not crop coordinates.

## Verifying the port

`codec.py` was differentially fuzzed against upstream's verbatim notebook
implementation: **30,000 random token streams, byte-identical output**, plus
20,000 exact `Ink -> tokens -> Ink` round-trips. Re-run the suite with
`uv run pytest`.

## Where the code lives

The split described above is implemented as:

- `ink_tokenizer/tf_runtime.py` — the only module importing TensorFlow. Holds
  the XLA shim and `InkSightEngine` (load + batch + fallback retry).
- `ink_tokenizer/local.py` — `LocalBackend`, in-process inference.
- `ink_tokenizer/client.py` — `InkSightClient`, the same API over HTTP.
- `services/inksight/app.py` — FastAPI over the *same* engine, so the service
  and in-process paths cannot drift apart.

`docs/running-inference.md` covers choosing between them.
