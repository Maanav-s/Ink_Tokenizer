# Content prompt: writing a synthetic whiteboard page

This is the prompt (and schema) for step 1 of the synthetic data pipeline
([synthetic_data.md](synthetic_data.md)): an LLM writes **what** is on a
whiteboard page as `content.json`. It never writes coordinates, sizes or
drawing order. `scripts/layout_engine.py` decides those, differently for
every layout seed, so one content file yields many distinct pages.

Run `scripts/validate_page.py` on every generated file (the layout engine
runs `check_content` too). It also prints warnings for content it could not
check, e.g. an equation with no `check` or a circuit that matches no truth
table. Reject files that fail rather than fixing them by
hand. That way the checks, not trust in the LLM, keep the ground truth right.

## Prompt

> You are writing the content of one engineer's whiteboard page from a
> design meeting or working session about **{topic}**. Output a single JSON
> object following the schema below, and nothing else.
>
> - Write like an engineer at a whiteboard: terse notes, abbreviations,
>   arrows (→), symbols (≤ ≥ ≈ ± μ Ω Δ), units, question marks. Not prose.
> - Include {mix}: for example a truth table, a table of action items or a
>   timeline, one or more equations or a short derivation, and a diagram
>   (flowchart, logic circuit or analog schematic). About 15–45 written
>   lines in total: a page, not a document.
> - Everything must be correct. Truth tables must be complete and consistent
>   with the expressions and circuits on the page. Equations must be right,
>   and continuation lines must follow from the line they continue. Add a
>   `check` with numeric values to equations that can be evaluated.
> - Use only the characters listed under "Characters". LaTeX only in `math`
>   elements and in `$...$` labels.
> - Give no positions, sizes or styling: the layout is generated separately.

## Schema (`ink_tokenizer.synthetic_page.content/v1`)

```json
{
  "schema": "ink_tokenizer.synthetic_page.content/v1",
  "page_id": "page_0005",
  "topic": "one line",
  "provenance": {"generator": "model name / prompt version", "automated": true},
  "elements": [ ... in reading order ... ]
}
```

Ids are letters, digits, `_` and `-`, unique across the page. Elements, in
reading order:

- **text**: `{"id", "type": "text", "role", "text"}`. `role` is `title`
  (one per page, first), `heading`, `bullet`, `subbullet`, `note`, `caption`
  or `text`. One line of writing. A heading is kept together with the
  element that follows it.
- **list**: `{"id", "type": "list", "style": "dash" | "numbered" | "plain",
  "heading"?: str, "items": [{"id"?, "text", "items"?: [...]}]}`. The text
  excludes the marker ("-", "1."): the layout writes markers. Nest up to 3
  levels. Give an item an `id` if an annotation refers to it.
- **table**: `{"id", "type": "table", "kind", "header"?: [str], "rows":
  [[str]]}`. Cells are strings (`""` for an empty cell) and may be
  `$...$` math. `kind` is `truth_table`, `action_items`, `timeline` or any
  short word. A truth table also has `input_columns` and `output_columns`
  (column indices that together cover every column), `row_order:
  "binary_counting_msb_first"` when its input rows count up in binary (all
  2^n rows), and only `0`, `1` or `x` cells. Rules (lines) are chosen by
  the layout.
- **math**: `{"id", "type": "math", "latex", "text", "defines"?,
  "continues"?, "check"?}`. One line of 2-D math in the LaTeX subset below.
  - `text` is a plain, checkable form: Python-like arithmetic (`*`, `/`,
    `**`, `sqrt`, `exp`, `abs`, `pi`, `j`), or a Boolean form (`+` or, `xor`,
    juxtaposition and, `'` not) using the truth table's column names.
  - `defines` names the quantity, or the truth-table column, that the line
    defines.
  - `continues` is the id of the line this one carries on from. Its `text`
    starts with `=`, and its `latex` with `=` or `\approx`; it is aligned
    under that line's relation.
  - `check: {"vars": {name: number or expression}, "rel_tol"?}` makes the
    validator evaluate every side of `text` and require them all to be
    equal. A side that is just the `defines` name is skipped unless `vars`
    gives it a value. Vars are evaluated in order and may be lambdas. A
    continuation also includes the previous line's last side and inherits
    that line's vars. `text` has no units, so keep each line's numbers in
    one consistent unit.
- **flowchart**: `{"id", "type": "flowchart", "direction"?: "TB" | "LR",
  "nodes": [{"id", "shape": "terminator" | "process" | "decision" | "io",
  "text"}], "edges": [{"id"?, "from", "to", "label"?}]}`. Keep node text
  short (≤ ~20 characters, `\n` for a line break). A decision has two
  out-edges, usually labelled `yes`/`no`.
- **circuit** (gate level): `{"id", "type": "circuit", "components":
  [{"id", "type", "label"?}], "wires": [{"id", "net", "from", "to"}],
  "labels"?: [{"id", "text", "attached_to": "net:<name>" | null}]}`.
  - Component types are `input_pin` (port `y`, needs `label`), `output_pin`
    (port `a`, needs `label`), `and2`, `or2`, `xor2`, `nand2`, `nor2`,
    `xnor2` (ports `a`, `b`, `y`) and `not`, `buf` (ports `a`, `y`).
  - One wire per driver-to-sink connection, written `"comp.port"`.
  - Pin labels that match a truth table's header are simulated against
    that table.
- **schematic** (analog): `{"id", "type": "schematic", "components":
  [{"id", "type", "label"?, "value"?, "at": [col, row], "orient"?: "h" |
  "v"}], "nets": [{"id", "connects": ["comp.port", ...]}]}`.
  - Types are `resistor`, `capacitor`, `inductor`, `diode`,
    `voltage_source`, `current_source` and `battery` (ports `a`, `b`; with
    `v`, `a` is on top, and with `h`, `a` is on the left; a source's `a` is
    +), plus `ground` and `terminal` (port `a`).
  - `at` is a coarse grid position, the one placement hint: lay the
    circuit out as you would sketch it, with grid positions not shared and
    neighbours one step apart.
- **annotation**: `{"id", "type": "annotation", "kind": "underline" |
  "box" | "circle" | "arrow", "target"}` (or `"from"`, `"to"` for an arrow).
  Targets are element ids, list item ids, or table rows as
  `<table>.r<i>`, where rows count from 0 and the header is not a row.
  The layout drops an arrow it cannot draw cleanly.

## Characters

Text can use printable ASCII, Greek letters (α–ω, Α–Ω, e.g. Ω μ Δ τ η)
and these symbols: `⊕ → ← ↔ ⇒ ≤ ≥ ≠ ≈ ± · × ∞ ∫ √ ∑ ∂ ∆ ° ∈`. Nothing else
(no •, ✓, ∇, ≡, emoji, curly quotes).

## LaTeX subset (math elements, `$...$` labels)

`^ _ {}`, `\frac`, `\sqrt` / `\sqrt[n]`, `\int` and `\sum` / `\prod` with
limits, `\left( \right)` (also `[ ] | .`), `\boxed`, `\overline`, `\bar`,
`\vec`, `\hat`, `\dot`, `\mathrm`, `\text`, `\operatorname`, function names
(`\sin`, `\log`, `\lim`, `\max`, ...), Greek letters, `\cdot \times \pm \le
\ge \ne \approx \infty \to \Rightarrow \leftrightarrow \oplus \in \partial
\circ \ldots \cdots \%`, and spacing `\, \; \quad`. One line per element:
no `\\`, `&`, matrices or `\mathcal` (see `scripts/math_typeset.py`).

## What makes good content for this project

The model trained on these pages suggests **structural** completions only
(AGENTS.md). Pages are most useful when they contain structure that a
writer would complete:
- truth tables;
- tables with repeated columns, or headers that count up (Wk41, Wk42, ...);
- numbered lists;
- flowchart decisions with yes/no branches;
- repeated diagram parts.

`scripts/targets.py` turns these into completion targets.
