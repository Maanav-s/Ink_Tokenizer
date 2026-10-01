# Synthetic training data

The training data for recognition and suggestion is synthesized rather than
collected. An LLM writes the content of whiteboard pages, and the pages are
then rendered as online ink (timed pen strokes), so every stroke comes with
the structure it belongs to.

## Why synthesize

- **Labels come for free.** The generator knows the structure it wrote down
  (the table, the equation, the diagram), so no one has to annotate ink by
  hand.
- **Partial input can be sampled.** Every stroke has a timestamp, so any
  prefix of a page is a valid "in-progress" example, with the rest of the page
  as the structural completion target. This matches the assumption that
  inputs are partial by definition (see [AGENTS.md](../AGENTS.md)).
- **Multiple writers can be simulated** by rendering different regions of a
  page in different handwriting styles and interleaving their strokes in
  time.
- **Coverage is controllable.** The mix of truth tables, math and diagrams,
  and how messy they are, is set by the generator instead of by whatever a
  collected dataset happens to contain.

## Pipeline

1. **Content generation (LLM).** An LLM writes what is on the page, and
   nothing about where: text, lists, tables, math as LaTeX, and diagrams as
   flowchart nodes and edges, gate netlists or schematics with coarse grid
   hints (`content.json`; prompt and schema in
   [content_prompt.md](content_prompt.md)). This is the ground truth for
   recognition and the source of completion targets.
   `scripts/validate_page.py` rejects content that is malformed or wrong:
   - characters no font can draw, or LaTeX that does not typeset;
   - truth tables that don't count in binary, or that disagree with the
     expressions defining their outputs or with a circuit that simulates
     them;
   - equations whose sides don't evaluate equal.
2. **Layout (code).** `scripts/layout_engine.py` places the content on the
   page from a layout seed, so one content file yields many pages. The seed
   decides:
   - text sizes and spacing, columns, and what goes beside what;
   - table rule styles, the order a table is filled in (by row, by
     column, inputs before outputs), and whether boxes or text come first
     in a flowchart;
   - diagram placement and routing (`scripts/diagram_layout.py`) and math
     typesetting (`scripts/math_typeset.py`);
   - the writing process: an occasional mistake that is struck through,
     scribbled out or overwritten, and sometimes a second writer working on
     another part of the page at the same time.

   Text boxes are sized with the widest font in the pool. A layout is
   accepted only if its clean ink, in every pool font, has no text touching
   other ink. `scripts/targets.py` records the structural completion
   targets the page offers (e.g. the remaining truth-table rows once two
   are written).
3. **Text to ink.** Handwritten text is produced by a handwriting synthesis
   model that turns a string into a stroke sequence. For now,
   single-stroke fonts with layered handwriting-like noise stand in for it
   (`scripts/fonts.py`, `scripts/hand_noise.py`).
4. **Diagrams to ink.** Diagrams are drawn from their geometric shapes
   (`scripts/shapes.py`), with heuristics that make them look hand-drawn:
   - wobbly, bowed lines;
   - overshoot and gaps at corners and joins;
   - uneven stroke speed;
   - variation in size and alignment.

   Labels inside diagrams go through the text-to-ink model.
5. **Page assembly.** `scripts/render_inkml.py` merges text and diagram
   strokes into one timed InkML stroke stream, each writer on their own
   clock. A capture-device model (`scripts/digitizer.py`) adds sample rate,
   hooks, dropped samples and quantization. Every stroke keeps a reference
   to its item and to the character or shape part it draws. Mistakes stay
   in the ink, labelled as deleted.

## Text-to-ink model

**Initial plan:** use an existing handwriting synthesis implementation such as
[pytorch-handwriting-synthesis-toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit)
to generate the handwritten text samples.

**Possible later plan:** train our own Mamba (state-space model) text-to-ink
model for the same job. This would give us control over the character set,
style conditioning and sampling behaviour. It is also a chance to build
experience with an SSM architecture on ink sequences before relying on one
elsewhere in the project.

Either way the rest of the pipeline treats the model as a black box: string
(plus style) in, timed strokes out. Keep that interface narrow so the model
can be swapped without touching layout or page assembly.

## Known risks

- **Synthetic-to-real gap.** A model trained only on synthetic pages may
  learn artefacts of the renderer (too-regular jitter, one model's
  handwriting quirks) instead of real handwriting. Evaluation, and the
  false-suggestion rate in particular, must be measured on **real** held-out
  ink, never only on synthetic pages.
- **Character coverage.** Handwriting synthesis models are usually trained on
  English text datasets and may not cover math symbols, Greek letters,
  subscripts or superscripts, or logic notation. Check what the chosen
  model supports before generating math. Unsupported symbols need another
  source (e.g. the diagram-style primitive renderer, or our own model).
- **LLM content quality.** Generated truth tables and equations can be
  wrong. The structure is the ground truth for training, so it should be
  validated programmatically where possible (e.g. recompute truth tables)
  rather than trusted as written.
- **Scope.** The generator produces content, but the models trained on it
  must still only predict *structure*. Completion targets taken from
  synthetic pages must follow the scope boundary in
  [AGENTS.md](../AGENTS.md).
