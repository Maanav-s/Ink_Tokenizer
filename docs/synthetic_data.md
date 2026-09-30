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

1. **Content generation (LLM).** An LLM produces whiteboard-style notes and
   diagrams in a structured form: the text and math to be written, tables as
   rows and cells, and diagrams as primitives (boxes, gates, wires, arrows,
   labels) with their connections. This structured form is the ground truth
   for recognition and the source of completion targets.
2. **Layout.** The content is placed on a page: positions, line spacing,
   table grids, diagram placement, and the order in which things are written.
3. **Text to ink.** Handwritten text is produced by a handwriting synthesis
   model that turns a string into a stroke sequence.
4. **Diagrams to ink.** Diagrams are drawn from their geometric primitives,
   with heuristics that make them look hand-drawn: artificial jitter on
   points, wobbly lines, overshoot and gaps at corners and joins, uneven
   stroke speed, and variation in size and alignment. Labels inside diagrams
   go through the text-to-ink model.
5. **Page assembly.** Text and diagram strokes are merged into a single timed
   stroke stream per page. Each stroke keeps a reference back to the
   structural element it came from.

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
