# Synthetic page artifacts

These are outputs of steps 1 (content generation) and 2 (layout) of the
synthetic data pipeline in [docs/synthetic_data.md](../docs/synthetic_data.md).
No ink has been rendered. Rendering (text-to-ink, diagram jitter, page
assembly) is a later stage that consumes these files.

**Provenance:** `page_0001` was written by hand by an LLM agent (Claude). No
automated generator exists yet. Its truth table, expressions and circuit were
checked by a script: the table was recomputed from arithmetic, each
expression was evaluated on all 8 input rows, and the netlist was simulated.
The layout was checked for page bounds, overlaps and wire routing.
Treat it as a sample of the format, not as generator output.

`scripts/make_sample_page.py` regenerates this page (its content is
hard-coded in the script), and `scripts/check_page.py [page_dir]` runs the
checks above.

`page.inkml` is the page rendered as ink by `scripts/render_inkml.py`, and
`page.png` is that InkML rasterized by `scripts/inkml_to_png.py` for
inspection. See "page.inkml" below.

Each page directory holds `content.json` (what is written, the ground truth)
and `layout.json` (where and in what order it is written). Both files key
everything by stable `id`s.

## content.json (`ink_tokenizer.synthetic_page.content/v0`)

Top level: `schema`, `page_id`, `topic`, `provenance`, `elements[]`.
Element types:

- `text` with `id`, `role` (`title` | `bullet`) and `text`.
- `truth_table` with:
  - `header[]` of `{id, text}`.
  - `rows[]` of `{id, cells[] of {id, text}}`.
  - `input_columns` and `output_columns` (column indices).
  - `row_order`, e.g. `binary_counting_msb_first`.
  - `rules[]`: the drawn lines. Each is `{id, kind: horizontal|vertical, after_row|after_column}`.
- `math` with:
  - `text`: a plain ASCII form, machine-checkable, using `xor` and implicit AND.
  - `written`: what is actually handwritten, which may include `⊕`.
  - `latex`.
  - `defines`: the signal the line defines.
  - Optionally `continues`, the id of the line it carries on from.
- `circuit` with:
  - `components[]` of `{id, type, label, ports[]}`. `type` is one of
    `input_pin`, `output_pin`, `and2`, `or2` or `xor2`. Gates have ports
    `a`, `b` and `y`, input pins have `y`, and output pins have `a`.
  - `wires[]` of `{id, net, from: "comp.port", to: "comp.port"}`. Each is one
    point-to-point connection, and a net with fan-out has one wire per sink.
  - `labels[]` of `{id, text, attached_to}`, where `attached_to` is `net:<name>` or `null`.
  - `implements`: maps each output to the math element that describes it.

## layout.json (`ink_tokenizer.synthetic_page.layout/v0`)

- `unit`: `mm`. `page` is `{width: 1200, height: 800, origin: top-left, y_axis: down}`.
- `writers[]`: a list of `{id, style}`. Every placed item carries a `writer`
  field, so a page can have several writers later. This page has one,
  `writer_0`.
- `text_metrics.char_advance_per_height`: the glyph-width estimate used to
  size text boxes (0.55 × text height). The renderer may refit the boxes to
  the ink it actually produces.
- `elements`: placement keyed by top-level content id. Every box is
  `[x0, y0, x1, y1]`.
  - Text and math: `bbox` and `text_height`.
  - Table: `bbox`, plus `column_x[]` and `row_y[]` for the grid boundaries.
    `cells{id: {cell_box, text_bbox, row, col}}` gives each cell, and `row`
    is `"header"` for the header. `rules{id: {polyline}}` gives the drawn lines.
  - Circuit: `bbox` and `components{id: {bbox, ports{name: [x, y]}}}`. Pins
    also have `label_bbox` and `label_text_height`. `wires{id: {polyline: [[x, y], ...]}}`
    holds orthogonal polylines from the source port to the sink port.
    `junctions[]` gives the fan-out dots as `{net, point}`, and
    `labels{id: {bbox, text_height}}` places the labels.
- `writing_order`: the order in which things are written. Each entry is a
  leaf id: a text line, a math line, a table cell or rule, a component, a wire
  or a label. Container ids (`tt_1`, `dia_1`) never appear. On this page, the
  table's input cells are filled row by row, then the `S` column and then the
  `Cout` column are filled top to bottom.

## Notes for the rendering stages

- Gate shapes come from `type`, and the gate's `bbox` is the space the
  symbol must fit in.
- Wire crossings that have no junction dot are not connections.
- `math.written` uses `⊕`. The handwriting synthesis model may not support
  that glyph (see "Character coverage" in the pipeline doc). If so, draw it
  as a primitive.
- Completion targets taken from this page must stay structural. Examples
  are the remaining truth-table rows and repeated gate scaffolding (see
  AGENTS.md).

## page.inkml (naive Hershey rendering)

A proof of concept, not training-quality ink. `scripts/render_inkml.py`
draws text with a single-stroke Hershey font (`futural` by default),
draws `⊕` as a circle plus a cross, and draws diagrams from plain geometric
primitives. It adds no jitter and uses no handwriting model.

- `traceFormat` has the channels `X`, `Y` (mm, same frame as layout.json) and
  `T` (seconds from the first stroke).
- Elements follow `writing_order`. Strokes are sampled at 100 Hz with a
  constant pen speed of 150 mm/s. Pen-up travel and a pause between elements
  add the gaps in `T`.
- `<annotation>`s on `<ink>` give `page_id`, the page size and the y-axis
  direction.
- The top-level `<traceGroup xml:id="elements">` holds one child
  `<traceGroup>` per `writing_order` entry. Each child has an `element_id`
  and a `writer` annotation, and `traceView`s that point to its traces.
  Junction dots are grouped with the wire that branches at them.

Commands (run with `uv sync --extra render` installed):

```bash
uv run --extra render python scripts/render_inkml.py [page_dir] [--font NAME]
uv run --extra render python scripts/inkml_to_png.py artifacts/page_0001/page.inkml \
    [out.png] [--until SECONDS] [--color-groups]
```

`--until` draws only the ink written up to that time, i.e. an in-progress
page. `--color-groups` colours each element separately.
