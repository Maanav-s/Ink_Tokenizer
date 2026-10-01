# Synthetic page artifacts

Each `page_NNNN/` directory holds one synthetic whiteboard page, made by the
pipeline in [docs/synthetic_data.md](../docs/synthetic_data.md):

| File | Made by | What it is |
|---|---|---|
| `content.json` | an LLM ([docs/content_prompt.md](../docs/content_prompt.md)) | what is written: text, lists, tables, math, diagrams. No coordinates. The ground truth. |
| `layout.json` | `scripts/layout_engine.py --seed 0` | where and in what order everything is written, by whom, with which mistakes; plus completion targets |
| `layouts/layout_s<N>.json` | `layout_engine.py --seed N` | other layouts of the same content |
| `page.inkml`, `page.png` | `scripts/render_inkml.py` | layout seed 0 rendered as clean ink (futural font, no noise) |
| `renders/L<layout>_<fonts>_noise<L>_seed<N>.inkml` | `scripts/render_batch.py` | handwriting-like renders, with `.png` previews and `.targets.json` sidecars |
| `renders/manifest.json` | `render_batch.py` | the renders kept, and those rejected by the ink check |

Only `content.json` and the layouts are committed. Rendered ink
(`page.*`, `renders/`) is generated locally and kept out of git to save
storage; recreate it with `render_inkml.py` and `render_batch.py` (see
Commands).

## Pages

| Page | Content |
|---|---|
| page_0001 | 1-bit full adder: notes, truth table, Boolean algebra, gate-level circuit |
| page_0002 | CI deploy pipeline flowchart with canary and rollback |
| page_0003 | dense design-review notes: lists, action-item and timeline tables, annotations |
| page_0004 | RC low-pass derivation: fractions, sub/superscripts, ∫, √, schematic |
| page_0005 | 2:1 mux and 4-bit parity design review: truth table, Boolean algebra, gate circuit, numbered list |
| page_0006 | payments API sprint planning + incident retro: dense lists, action-item and timeline tables, retry flowchart |
| page_0007 | sensor-node power budget and LED sizing: derivation with checks, schematic, current table |

**Provenance:**
- **Pages 1–4:** an LLM agent (Claude) wrote their content by hand. They
  began as hand-placed v0 pages (content and coordinates together) and were
  ported to content-only v1.
- **Pages 5–7:** written by LLM subagents given only
  [docs/content_prompt.md](../docs/content_prompt.md) and a topic, then
  checked with `validate_page.py`. All three passed on the first try.
- **Checks:** every page's truth tables, Boolean expressions, circuits and
  numeric equations are checked by `scripts/validate_page.py`.

Pages 6 and 7 hold more than fits on the standard 1200 × 800 mm board, so
the engine gave them a larger one (1500 × 1000 or 1800 × 1200 mm). No
layout of page 6 fits with seed 2.

## content.json (`ink_tokenizer.synthetic_page.content/v1`)

The schema, and the prompt that produces it, are in
[docs/content_prompt.md](../docs/content_prompt.md). The element types are:

- `text`, `list` and `table`;
- `math`: LaTeX, plus a checkable plain form;
- `flowchart`;
- `circuit` (gate level);
- `schematic` (analog, with coarse grid hints);
- `annotation`: underline, box, circle or arrow.

## layout.json (`ink_tokenizer.synthetic_page.layout/v1`)

- `page`: `{width: 1200, height: 800, origin: top-left, y_axis: down}`, in
  mm (`unit`).
- `layout_seed`, and `style`: the drawn style parameters, such as text
  heights, spacing, table style and fill order, whether there is a second
  writer, and the mistake rate.
- `writers`: `[{id, start_after?}]`. A second writer starts when the first
  finishes the item `start_after`, and from then on both write at once.
- `items`: `{leaf id: item}`, one per entry of `writing_order`. Each item
  has these fields:
  - `kind` and its geometry (below);
  - `writer`;
  - `element`: the content element it belongs to;
  - `role`, e.g. `title`, `bullet`, `list_marker`, `header_cell`, `cell`,
    `rule`, `math`, `node_box`, `edge`, `gate`, `wire` or
    `annotation_box`;
  - optional `bold`;
  - table cells also have `row` and `col`.
- `writing_order`: the order items are written. Strokes are timed per
  writer, so the two writers' strokes interleave.
- `targets`: completion targets (below).

Geometry is stored only in the keys `bbox` (`[x0, y0, x1, y1]`), `points`,
`point`, `a`, `b`, `baseline` and `dots`. Item kinds:

| kind | fields | drawn as |
|---|---|---|
| `text` | `text`, `bbox`, `text_height`, `align`?, `corrections`? | one line in a single-stroke font. The bbox bottom is the descender line, and capitals reach `text_height` above it. Squeezed (to at least 75%) if wider than the bbox. |
| `math` | `latex`, `bbox`, `baseline`, `text_height`, `options`? | typeset by `scripts/math_typeset.py`. `options`: `frac_order` (`num_bar_den` / `bar_num_den`), `radical_bar` (`joined` / `after`) |
| `polyline` | `points`, `closed`? | a hand-drawn line |
| `rect` | `bbox`, `corner_radius`? | |
| `ellipse`, `diamond` | `bbox` | |
| `parallelogram` | `bbox`, `skew` | |
| `arrow` | `points` (shaft, ending at the tip), `head`? | |
| `wire` | `points`, `dots`? | a circuit wire, plus junction dots drawn with it |
| `dot` | `point` | junction dot |
| `pin` | `point`, `radius` | circuit pin (open circle) |
| `gate` | `gate`, `bbox` | and2, or2, xor2, nand2, nor2, xnor2, not, buf. Ports from `shapes.gate_ports()` |
| `component` | `component`, `a`, `b`?, `style`?, `size`? | resistor, capacitor, inductor, diode, voltage_source, current_source, battery (from a to b); ground, terminal (at a) |

`corrections` on a text item lists the writer's mistakes, as
`{index, wrong, style}`. The writer first wrote `wrong` where
`text[index:]` starts:
- `strike` and `scribble` cross it out, and the right text continues to
  its right (the bbox includes the extra width);
- `overwrite` writes the right characters over the wrong ones.

`text` stays the correct string.

## Completion targets

`targets` (from `scripts/targets.py`) lists structural completions:
`{id, kind, tier, element, after: {id}, target: [{id}], ...}`. Once the
page is written up to the end of item `after`, the `target` items are a
correct completion. The kinds are:

| kind | meaning |
|---|---|
| `truth_table_inputs` | input cells in binary counting order. `span: next` is the next run; `remaining` is all of them |
| `truth_table_outputs` | output cells, once their defining expression is on the page. Tier `arithmetic` |
| `table_rules` | the rest of a table's rules after the first |
| `header_sequence` | the rest of a header that counts up (Wk41, Wk42, ...) |
| `list_marker` | the next marker of a numbered list. `speculative`: the list might end |
| `branch_label` | the other label of a yes/no decision |
| `node_outline` | a flowchart node's box, when its text was written first |

Each render's `.targets.json` resolves them to `t_context_end` (seconds)
and `target_traces`. A target is dropped when another writer's strokes
break its time order.

## Rendered ink (InkML)

`scripts/render_inkml.py` draws text with single-stroke fonts and diagrams
from shapes, in `writing_order`. `--noise LEVEL` adds handwriting-like
variation from `scripts/hand_noise.py`, layered as follows:

- **Writer:** slant, size and pen speed.
- **Line:** offset, tilt and curvature. The tilt is capped so a long line
  drifts at most 0.3 h, and 0.12 h in a table cell.
- **Character shape (allograph):** each writer keeps a consistent version
  of each character.
- **Character instance:** each occurrence varies slightly.
- **Stroke:**
  - overshoot and endpoint error;
  - bowed lines (the bow grows with length up to 150 mm);
  - imperfect circles;
  - pen speed and pauses.

`scripts/digitizer.py` adds capture artefacts:
- the sample rate (60–200 Hz);
- pen-down and pen-up hooks;
- dropped samples and bursts of them;
- coordinate quantization;
- timestamp jitter.

The batch renders use noise 0.6. Level 0 is clean geometry on an ideal
100 Hz device, and the same (layout, fonts, level, seed) always gives the
same file.

- `traceFormat`: `X`, `Y` in mm (layout frame) and `T` in seconds. Traces
  are numbered in time order, so with two writers they interleave.
- Brushes: `pen` (2 mm) and `bold` (3.5 mm, titles).
- `<annotation>`s on `<ink>` give:
  - `page_id` and the page size;
  - `layout_seed`;
  - `writer_fonts`, e.g. `writer_0:cursive,writer_1:EMSTech`;
  - `noise_level` and `noise_seed`;
  - `device`, the digitizer profile.
- Labels:
  - `<traceGroup xml:id="elements">` holds one `<traceGroup>` per item,
    with annotations `element_id`, `element`, `writer`, `kind`, `role`, and
    `truth` (the item's text or LaTeX).
  - Each item group holds one `<traceGroup>` per character or shape part.
    It is annotated with `truth` (the character, or a part name such as
    `rect`, `head`, `frac_bar`, `junction`), `char_index` (the position in
    the item's text, for text), `src` (the offset in the LaTeX, for math),
    and `status`: `deleted` for a mistaken character, `mark` for a
    strike-through or scribble.

## Commands

```bash
uv sync --extra render
uv run --extra render python scripts/layout_engine.py artifacts/page_0001 [--seed N]
uv run --extra render python scripts/validate_page.py artifacts/page_0001 [--layout FILE]
uv run --extra render python scripts/render_inkml.py artifacts/page_0001 [--layout FILE] \
    [--font F[,F2]] [--noise 0.6] [--seed N] [--out FILE] [--no-device]
uv run --extra render python scripts/inkml_to_png.py page.inkml [out.png] \
    [--until SECONDS] [--color-groups] [--crop X0 Y0 X1 Y1] [--px-per-unit N]
uv run --extra render python scripts/render_batch.py [page_dir ...] \
    [--layouts 0,1,2] [--fonts ...] [--seeds 0] [--noise 0.6] [--relayout]
```

`--until` draws only the ink written by then, i.e. an in-progress page.
