# Running inference

Two backends, same API. Pick by where the model lives relative to you.

| | `LocalBackend` | `InkSightClient` |
|---|---|---|
| Runs | in-process | over HTTP |
| Needs | TensorFlow (Linux/macOS) | nothing special |
| Use when | you are *on* the GPU machine | the model is elsewhere |

Both subclass `DerenderBackend`, so `derender_image`, `derender_boxes`,
`derender_crops` and `merge` behave identically and swapping one for the other
changes no coordinates. `get_backend()` picks for you: in-process if
TensorFlow imports, HTTP otherwise.

## On a Linux machine (the normal case)

```bash
git clone <this repo> && cd Ink_Tokenizer
./scripts/setup_linux.sh --prefetch
```

That installs `uv` into `~/.local/bin` if it is missing — **no root needed** —
lets uv fetch a suitable Python, installs the inference extra, runs the tests,
downloads the 692 MB checkpoint, and prints a readiness report. Re-runnable.

If you would rather use the system Python and pip, pass `--no-uv`; you will
need `python3-venv` and a Python in `>=3.11,<3.14`.

Then:

```bash
uv run ink-tokenizer doctor                 # what this machine can do
uv run ink-tokenizer derender word.jpg      # straight to ink
uv run ink-tokenizer derender *.jpg -o ink.json
```

or from Python:

```python
from PIL import Image
from ink_tokenizer import LocalBackend

with LocalBackend() as backend:                  # loads the model once
    result = backend.derender_image(Image.open("word.jpg"))
    print(result.text, result.ink.num_points)
```

Reuse one `LocalBackend` across a whole dataset job — construction loads a
692 MB graph, and batching is what makes bulk derendering affordable.

### GPU — get one for bulk work

CPU works, but not at dataset scale. InkSight decodes autoregressively: one
word's ink is hundreds of sequential decoder steps. Measured end-to-end on a
CPU-only box (WSL2, no AVX-512, 250x126 input): **11.5 s to load the graph,
then 181 s for a single word** — 11 strokes, 145 points. At that rate a
40-word page is over two hours and a corpus is not worth starting.

Use the GPU box for anything past a smoke test.

For GPU, change the tensorflow pin in `pyproject.toml` to
`tensorflow[and-cuda]==2.20.0` and re-sync. `ink-tokenizer doctor` prints the
visible devices, so you can confirm the card is actually being used rather
than discovering it was CPU all along after an overnight job.

Two further things matter for throughput:

- **Batch.** `derender_crops` and `derender_boxes` send whole batches
  (default 32); calling `derender_image` in a loop pays per-call overhead and
  wastes the GPU.
- **Reuse the backend.** Constructing `LocalBackend` loads a 692 MB graph.
  Build it once per job, not once per image.

Both `LocalBackend` and the service set GPU memory growth rather than
reserving the whole card, which matters on a shared lab box.

### Air-gapped machines

`--model` accepts a local SavedModel directory, so you can copy a prefetched
checkpoint across and skip the Hugging Face download:

```bash
ink-tokenizer derender word.jpg --model /shared/InkSight-Small-p
```

## From Windows (or any machine without TensorFlow)

`tensorflow-text` publishes no Windows wheels, so the model cannot run
in-process there. `uv sync --extra inference` is a deliberate no-op on Windows
rather than an error — the codec, geometry and tests all still work, and the
test suite runs without TensorFlow at all.

To actually derender, run the service on a Linux box:

```bash
uv run uvicorn app:app --app-dir services/inksight --host 0.0.0.0 --port 8000
```

and point a client at it:

```python
from ink_tokenizer import InkSightClient

with InkSightClient("http://lab-box:8000") as backend:
    print(backend.health())
    result = backend.derender_image(Image.open("word.jpg"))
```

The CLI takes `--endpoint` for the same thing:

```bash
ink-tokenizer derender word.jpg --endpoint http://lab-box:8000
```

Docker is also available (`services/inksight/compose.yaml`) if you want the
pinned environment rather than the host's. Build context is the repo root,
because the service installs `ink_tokenizer` and shares its engine.

## What crosses the wire

Only `{"text": ..., "used_fallback": ...}` — raw model output. Token decoding
and the inverse pad transform happen client-side, so a service upgrade cannot
silently change your coordinates, and every geometry bug is reproducible
without a GPU. See [inksight.md](inksight.md) for why the split exists.
