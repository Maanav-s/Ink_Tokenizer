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

### Training Data

Training data is **synthesized**. An LLM generates whiteboard notes and
diagrams in a structured form. A text-to-ink model renders the text as
handwriting, and heuristics such as artificial jitter make the diagrams look
hand-drawn. Together they produce timed stroke pages whose structure labels
come for free. The first text-to-ink model will be an existing handwriting
synthesis implementation
([pytorch-handwriting-synthesis-toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit)).
A self-trained Mamba SSM may replace it later. See
[docs/synthetic_data.md](docs/synthetic_data.md) for the pipeline and its
risks. Most importantly, evaluation must still use real handwriting.

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

Early scaffold — Python 3.13, managed with `uv`. The only dependencies are
the optional `render` extra (Hershey-Fonts, Pillow).

- [pyproject.toml](pyproject.toml) — project metadata
- [main.py](main.py) — placeholder entry point
- [docs/synthetic_data.md](docs/synthetic_data.md) — synthetic training data
  plan
- [docs/slurm.md](docs/slurm.md) — running jobs on Slurm
- [artifacts/](artifacts/README.md) — sample synthetic pages: content,
  layouts, clean and handwriting-like InkML renders. `artifacts/` is
  gitignored, so new pages need `git add -f`.
- Synthetic page pipeline (in `scripts/`; needs `uv sync --extra render`):
  - [docs/content_prompt.md](docs/content_prompt.md) — prompt and schema for
    LLM-written `content.json` (content only, no coordinates).
  - [validate_page.py](scripts/validate_page.py) — content checks (truth
    tables, Boolean expressions, circuit simulation, numeric equations),
    layout checks, and ink-overlap checks on rendered ink.
  - [layout_engine.py](scripts/layout_engine.py) — content -> layout.json
    from a layout seed: placement, styles, writing order, mistakes, a second
    writer. It uses [diagram_layout.py](scripts/diagram_layout.py)
    (flowcharts, circuits, schematics with orthogonal routing),
    [math_typeset.py](scripts/math_typeset.py) (LaTeX subset) and
    [targets.py](scripts/targets.py) (structural completion targets).
  - [render_inkml.py](scripts/render_inkml.py) — layout -> InkML, labelled
    per item and per character or shape part. It uses
    [fonts.py](scripts/fonts.py) (single-stroke fonts, pool metrics),
    [shapes.py](scripts/shapes.py),
    [hand_noise.py](scripts/hand_noise.py) (handwriting-like variation at
    the writer, line, character and stroke levels, rather than i.i.d.
    Gaussian noise) and [digitizer.py](scripts/digitizer.py) (capture
    artefacts).
  - [render_batch.py](scripts/render_batch.py) — layouts x fonts x seeds,
    rejecting renders whose text ink collides.
  - [inkml_to_png.py](scripts/inkml_to_png.py) — rasterize InkML for
    inspection.
  - [fonts/](fonts/README.md) holds OFL-licensed single-line SVG fonts.
  - `scripts/legacy_v0/` — generators for the old hand-placed v0 pages,
    kept for reference; the current renderer does not read v0 layouts.
- Ink format: **InkML**, with X/Y in mm and T in seconds.
- Text-to-ink teacher (needs the `model` extra):
  - [teacher_model.py](scripts/teacher_model.py) — the pretrained Graves
    synthesis network from pytorch-handwriting-synthesis-toolkit, ported to
    current torch (verified to sample identically to the original) with
    batched sampling and per-point character alignment.
  - [fetch_teacher.py](scripts/fetch_teacher.py) — downloads its pinned
    checkpoint into `models/teacher/` (gitignored).
  - [text_to_ink.py](scripts/text_to_ink.py) — the text + style -> labelled
    strokes interface (writing order, no timestamps). It rejects lines and
    priming that are clearly broken. It cannot detect misspelled words,
    which the teacher produces often.
- Mamba text-to-ink student, distilled from the teacher (bootstrap until
  IAM-OnDB access arrives; it inherits the teacher's misspellings):
  - [make_corpus_lines.py](scripts/make_corpus_lines.py) — the line list
    (dictionary prose + charset-limited notation + page strings). Stdlib
    only; run it on the host, which has the word list.
  - [generate_teacher_corpus.py](scripts/generate_teacher_corpus.py) —
    teacher samples as `.npz` shards; resumable and splittable across jobs.
    About 20-25% of the teacher's attempts are rejected.
  - [generate_corpus.sh](scripts/generate_corpus.sh) — runs it on a list of
    GPUs, one job per GPU over a split shard range, then pushes once.
  - [corpus_artifact.py](scripts/corpus_artifact.py) — keeps the shards in
    a W&B dataset artifact (`push`/`pull`). Generation, training and eval
    take `--wandb-entity`, `--wandb-project` and `--artifact`. Training
    records which artifact version it used and trains on exactly those
    shards.
  - [ink_corpus.py](scripts/ink_corpus.py) — model-agnostic shard loader and
    length-bucketed batching (reusable by other student architectures).
  - [student_model.py](scripts/student_model.py) — Mamba2 decoder over
    `[text, SEP, ink]`, no attention window: a mixture-density head for the
    next point and a character-index head that labels points and decides
    when the line is finished.
  - [train_student.py](scripts/train_student.py) — training; resumes from
    its run directory. Every `--sample-every` steps it draws held-out lines
    as written by the teacher and by the student, alternating rows, into
    `<run-dir>/samples/` and W&B.
  - [sample_student.py](scripts/sample_student.py) — preview, and a failure
    rate by line length next to the teacher's rejection rate.
- Environment: an Apptainer image (Python 3.13 + uv + gcc) with no Python
  packages baked in. `run_in_apptainer.sh` syncs `.venv` to `uv.lock` inside
  it. The `model` extra pins torch 2.9 and prebuilt CUDA 12 wheels of
  mamba-ssm / causal-conv1d from GitHub releases; bump them together.

Common commands:

```bash
scripts/build_apptainer_image.sh                  # once: .apptainer/ink_tokenizer.sif
scripts/run_in_apptainer.sh cpu python scripts/fetch_teacher.py
scripts/run_in_apptainer.sh cpu python scripts/text_to_ink.py "Hello world" --out hello.png
scripts/run_in_apptainer.sh 0 <command>           # with GPU 0

# Student pipeline (generation and training belong on a cluster). W&B flags
# are optional; without --wandb-project everything stays local.
WB="--wandb-entity <entity> --wandb-project <project>"
python3 scripts/make_corpus_lines.py              # host: data/teacher_corpus/lines.txt
scripts/generate_corpus.sh 0,1,2,3 0 64 $WB          # shards 0-63 split over GPUs 0-3, one push at the end
scripts/run_in_apptainer.sh cpu python scripts/corpus_artifact.py push $WB   # e.g. after a race
scripts/run_in_apptainer.sh cpu python scripts/corpus_artifact.py pull $WB [--artifact teacher_corpus:v3]
scripts/run_in_apptainer.sh 0 python scripts/train_student.py --run-dir models/student/<name> $WB
scripts/run_in_apptainer.sh 0 python scripts/sample_student.py \
    --checkpoint models/student/<name>/checkpoint.pt --eval 1000 $WB
```

Give each cluster or job its own `--first-shard` range; a shard index
generated in two places has the same name, and the artifact keeps one copy.
W&B credentials come from `WANDB_API_KEY` or `wandb login` (`~/.netrc`),
which the container inherits. W&B's run files and cache go in `.cache/wandb`.

On tacc, wrap the same commands in `scripts/submit_slurm.sh --module
tacc-apptainer/1.4.1 --` (see docs/slurm.md), build the image and run one
sync (`run_in_apptainer.sh cpu true`) on the login node first, and set
`INK_NO_SYNC=1` for the jobs. Compute nodes need outbound internet for W&B. Mamba2's Triton kernels compile on first use
(about a minute) into `.cache/triton`.

Update this section as real structure lands (synthetic data pipeline,
recognition model, suggestion model, evaluation harness).

## Conventions

- Keep the scope boundary above visible in code and docs: name modules for what
  they do structurally, not for "solving" anything.
- Evaluation should report the false-suggestion rate separately from accuracy;
  an aggregate number hides the metric that actually matters here.
