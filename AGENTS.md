# Ink Tokenizer — Agent Guide

Read this file before making changes. It defines what this project is for and,
just as importantly, what it deliberately will not do.

## Project Objective

### What We're Building

An AI model that recognizes engineering work as it's being handwritten on a
whiteboard (or digital tablet) and autocompletes the **structural** next step
inline — the way code autocomplete works, but for handwriting and diagrams.

### Scope

- **Context:** engineers using a whiteboard in a meeting/design session
- **Input:** handwritten math, truth tables, and circuit/block diagrams
- **Output:** inline, low-confidence-tolerant suggestions limited to:
  - Structural completion (e.g., remaining truth table rows, repeated diagram
    scaffolding, table/timeline headers)
  - Simple, near-zero-error-rate arithmetic (tentative — may be cut)
- **Explicitly out of scope:** solving equations/integrals, generating graphs,
  or predicting diagram/design *content* (anything where a wrong suggestion
  could mislead rather than just save time)

If a proposed feature predicts *content* rather than *structure*, it is out of
scope by default. Raise it rather than building it.

### Core ML Problems to Solve

1. **Recognition** — parse messy, in-progress handwritten math, tables, and
   diagrams into a structured representation, in real time, on a
   shared/multi-writer surface
2. **Suggestion generation** — given partial structure, predict the correct
   structural completion with very high precision
3. **Confidence gating** — suppress suggestions below a strict confidence
   threshold; a visible wrong suggestion in a live meeting is worse than no
   suggestion at all

### Success Criteria for the ML Component

- Near-zero rate of visibly incorrect suggestions (trust is the product)
- Low enough latency to feel like "autocomplete," not "chat response"
- Robust to messy handwriting, multiple writers, and partial/in-progress
  strokes

## What This Means in Practice

- **Precision over recall.** Showing nothing is an acceptable outcome. Showing
  something wrong is not. Default to suppressing a suggestion when unsure.
- **Confidence is a first-class output**, not an afterthought bolted on at the
  end. Anything that produces a suggestion should also produce a calibrated
  score that the gate can act on.
- **Latency is a correctness constraint.** A late suggestion is a wrong
  suggestion, because the writer has already moved on.
- **Inputs are partial by definition.** Strokes arrive mid-symbol, mid-row, and
  from more than one person at once. Code and models should assume incomplete,
  interleaved input rather than clean finished artifacts.

## Repository State

Python 3.11-3.13, managed with `uv`. Current work is **stage 1: data** --
bootstrapping digitized notes into online-ink-style stroke data using InkSight.

```
ink_tokenizer/          main package -- imports cleanly on ANY platform
  ink.py                Stroke / Ink data model + JSON
  codec.py              ink-token <-> Ink codec (ported from InkSight)
  preprocess.py         crop -> 224x224 padding and its exact inverse
  base.py               DerenderBackend: geometry + decoding, shared
  local.py              LocalBackend  -- in-process inference (Linux/macOS)
  client.py             InkSightClient -- same API over HTTP (anywhere)
  tf_runtime.py         the ONLY module that imports TensorFlow
  cli.py                `ink-tokenizer doctor` / `derender`
services/inksight/      FastAPI wrapper around tf_runtime, for remote callers
scripts/setup_linux.sh  one-shot bootstrap for a Linux/lab machine
docs/                   inksight.md (decisions, constraints)
                        running-inference.md (how to run it)
tests/                  codec + geometry tests; need no model and no TF
```

Commands:

```bash
uv sync --extra dev                 # main package, works on any platform
uv run pytest                       # no TF, no GPU, no model download
./scripts/setup_linux.sh --prefetch # Linux: full inference setup
uv run ink-tokenizer doctor         # what can this machine actually do?
```

### The TensorFlow boundary -- keep it

`tensorflow-text` publishes **no Windows wheels**, and InkSight pins TF 2.20.
So:

- `tf_runtime.py` is the only module that may `import tensorflow`, and it must
  configure `XLA_FLAGS` before doing so (the checkpoint predates TF 2.18's XLA
  change). Never `import tensorflow` anywhere else, and never above that shim.
- `ink_tokenizer/__init__.py` must stay importable without TensorFlow.
  `LocalBackend` is exposed through a module-level `__getattr__` for this
  reason -- do not add it to the eager imports.
- The `inference` extra is marked `sys_platform != 'win32'` so it resolves to
  nothing on Windows rather than failing.
- Coordinate math and token handling belong in the main package, where they
  are testable without a container. The service's only job is image bytes ->
  raw model text.

Read [docs/inksight.md](docs/inksight.md) before touching either side. It
covers why the split exists and the constraints InkSight imposes -- most
importantly that **its output has no timestamps** and that its page pipeline
assumes OCR word boxes, which fit text but not the math and diagrams this
project targets.

## Conventions

- Keep the scope boundary above visible in code and docs: name modules for what
  they do structurally, not for "solving" anything.
- Evaluation should report the false-suggestion rate separately from accuracy;
  an aggregate number hides the metric that actually matters here.
