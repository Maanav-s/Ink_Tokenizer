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

Early scaffold — Python 3.13, managed with `uv`, no dependencies yet.

- [pyproject.toml](pyproject.toml) — project metadata
- [main.py](main.py) — placeholder entry point

Common commands:

```bash
uv sync          # install/resolve dependencies
uv run main.py   # run the entry point
```

Update this section as real structure lands (data pipeline, recognition model,
suggestion model, evaluation harness).

## Conventions

- Keep the scope boundary above visible in code and docs: name modules for what
  they do structurally, not for "solving" anything.
- Evaluation should report the false-suggestion rate separately from accuracy;
  an aggregate number hides the metric that actually matters here.
