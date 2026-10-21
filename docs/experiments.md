# Experiments

A log of experiments that have been run and ones under consideration. Add an
entry when an experiment is planned, and fill in its result when it finishes.
Results should come from held-out data. For anything that produces suggestions,
report the false-suggestion rate separately from accuracy (see AGENTS.md).

Status values: `planned`, `running`, `done`, `dropped`.

## Past and running

### Text-to-ink students

| Experiment | Code | Status | Result |
|---|---|---|---|
| Mamba student distilled from the Graves teacher | `772edd9`, [student_model.py](../scripts/student_model.py) | done | |
| Residual cross-attention in the Mamba student | `e9a5fca` | done | |
| Mamba student on MathWriting human inks | `61c0edb`, `5ae86a6` | done | |
| `unpadded` vs `transformer` text conditioning | `611998a`, [text_conditioning.md](text_conditioning.md) | | |
| Flow-matching student (whole line at once) | `cd0ff51`, `f7c917a`, `02c0916`, [flow_model.py](../scripts/flow_model.py) | | |
| Reflow for few-step flow sampling | `913f2e6`, [train_reflow.py](../scripts/train_reflow.py) | | |
| Flow student on the mixed corpus (formulas + teacher text) | `b54fd7c`, [mixed_corpus.py](../scripts/mixed_corpus.py) | | |
| Mamba + attention hybrid sized like the flow student | `9bb9335`, [hybrid_model.py](../scripts/hybrid_model.py) | | |
| Font-rendered MathWriting labels (`\typeset{...}`) added to the flow corpus | `8443925`, `58ba14b`, [font_corpus.py](../scripts/font_corpus.py) | done | not yet written up |
| Ink-to-LaTeX recognizer, to judge generated ink | `5aa8364`, `5999c13`, `1025563`, [recognizer_model.py](../scripts/recognizer_model.py) | done (`transformer_v1`) | val token acc 0.97 (teacher-forced); most misreads are plausible readings of unclear handwriting, see below |

### Ink-to-LaTeX recognizer

Status: `done` (`models/recognizer/transformer_v1`, W&B project `inkDecoder`)

Built to measure legibility, and later to give the flow student a loss for
it. The flow student's samples drop and duplicate symbols (a `\binom` row
drawn with a missing digit and an extra pair of brackets): its loss compares
every point with the corpus ink, and nothing in it counts tokens. Which
token point i belongs to varies between writers, so while the ink is still
noisy a group of points can be split between two symbols or claimed by none.

[recognizer_model.py](../scripts/recognizer_model.py) (8.4M parameters):
- input is the flow student's own ink representation (centred (x, y) in
  ink heights, +-1 pen), so generated and partly denoised ink can be read
  without a conversion or a hard pen threshold, and stays differentiable;
- two stride-2 Conv1d layers (4x fewer positions), then a 6-layer
  transformer encoder (d=256, 4 heads);
- a 3-layer autoregressive decoder over the corpus's LaTeX tokens. Not CTC:
  math is not written in reading order (fraction bars first, scripts and
  closing brackets later). A CTC head on the encoder is an auxiliary loss
  (weight 0.2).

[train_recognizer.py](../scripts/train_recognizer.py) trains it on
`data/font_corpus` (`\typeset{label}` read as `label`) with a random
rotation, shear and stretch plus point jitter per ink, and validates on
MathWriting's human `valid` shard. Each evaluation also reads 512 inks
greedily and logs the token error rate and the exact-read rate.

Result: teacher-forced val token accuracy 0.97. Greedy-read misreads
([inspect_recognizer.py](../scripts/inspect_recognizer.py), which logs them
to W&B) are, with a few exceptions, reasonable readings of bad handwriting;
in some the reading is arguably better than the label. Good enough to use as
a judge. Record the greedy token error rate and exact rate here when
quoting it, since those are what apply to generated ink.

Not done yet: training on partly denoised ink, which the loss below needs,
and a second recognizer (other seed and stride) kept for evaluation only.

Next (`planned`): the same recognizer on `data/font_mixed_corpus`
(`font_corpus.py --base data/mixed_corpus`): human MathWriting, font
renders, and the teacher's plain text as `\text{...}`, so it can judge
mixed-corpus students' text lines too (`transformer_v1` has no `\text`
token, so the legibility loss and measurement skip those lines). Validation
scores human math (`val_*`), teacher text (`val_text_*`) and font
renders of the valid labels (`val_font_*`) separately, and draws each
source's 10 worst reads to `<run-dir>/reads/` and W&B. Check that
`val_` holds up against `transformer_v1`'s, since the text lines take
training steps from the math.

[visualize_flow.py](../scripts/visualize_flow.py) (`283df41`) renders a flow
checkpoint's sample grid at a chosen number of Euler steps and guidance, as
its own W&B run; it was used for a steps and guidance sweep.

## Planned

### Font-rendered labels for legibility

Status: `done` (results not yet written up)

The aim is a flow student that writes legibly and gets the LaTeX right
(layouts, every token drawn), rather than one whose ink looks more human.
MathWriting has no per-point alignment, so the token head never trains on
math, and each expression appears in only a few writers' hands.

[font_corpus.py](../scripts/font_corpus.py) typesets every unique MathWriting
train label (52,653) in EMSReadability and EMSTech with the page pipeline's
typesetter, and adds the 105,218 inks to the corpus alongside the 228k human
ones. Each rendered ink is stored as `\typeset{label}`, so the model can keep
the two hands apart and writes in the human hand when the wrapper is left out.
Every rendered point is labelled with the token it draws, so `char_ce` now
trains on math. Validation is MathWriting's human `valid` set only.

The typesetter was extended for this (`8443925`): matrices, every remaining
command in MathWriting's labels, and plain delimiters drawn as tall as what
they enclose. 99.97% of train labels typeset. Sample grids are in
`artifacts/font_latex/` (gitignored).

Run: continue `mathwriting_flow_height` on `data/font_corpus` with
`--init-from`. Baseline: the same checkpoint for the same number of steps on
`data/mathwriting_corpus` alone, so a gain cannot come from extra training.

Measure on human `valid` text, sampled without the wrapper:
- legibility: the recognizer's greedy token error rate on the samples
  (below), or a blind side-by-side of the two runs' sample grids;
- layout: fractions, scripts and matrices in the right places;
- that the human hand survives: samples should not drift towards the fonts.

Variants if it helps: `--base data/mixed_corpus` to keep the teacher's plain
text; `--splits train synthetic` to add MathWriting's ~396k synthetic labels;
hand noise ([hand_noise.py](../scripts/hand_noise.py)) on the renders.

### Fixing missing and duplicated symbols with the recognizer

Status: `planned`

In order of cost:
1. Measure: the recognizer's token error rate on student samples of human
   `valid` labels, for each flow checkpoint.
2. Best-of-N at sampling: draw N samples and keep one the recognizer reads
   correctly. No training change, and it shows how often the model can
   already write a label right.
3. A recognizer loss in training. The flow step already gives a one-step
   estimate of the finished ink, `x̂1 = xt + (1 − t)·v`; add
   `λ · CE(label | recognizer(x̂1))` with the recognizer frozen, weighted
   towards large t, where `x̂1` is sharp. Keep the flow loss dominant so the
   student cannot win by writing ink only the recognizer can read, and judge
   with the second recognizer. Needs the recognizer trained on partly
   denoised ink first. Code: [train_flow_legibility.py](../scripts/train_flow_legibility.py),
   which applies the term only at t in [0.8, 1] by default, in a second
   pass separate from the flow loss's times. Not yet run.
4. CFGRL with the recognizer as the optimality label: tag samples read
   correctly vs not, train with that tag as a condition, and guide towards
   "read correctly" (see the next section for the guidance form). No
   gradients through the recognizer.
5. If even best-of-N fails on repeated structure (`( )( )( )`): give the
   student an explicit plan of how many points each token gets, as TTS
   fixed skipped and repeated words, and condition each point on its token.
   Font ink has per-point token labels to train it; LaTeX is not written in
   token order, so the plan is an order over tokens, not just counts.

Measure: recognizer token error rate and exact rate on human `valid`
labels, and that human-likeness does not drop.

### Guiding away from font ink

Status: `planned` (after the font-rendered run; lower priority while
legibility is the goal)

Treat font ink as non-expert data and use classifier-free guidance to push
samples away from it, as CFGRL does for policies (Frans et al., "Diffusion
Guidance Is a Controllable Policy Improvement Operator", arXiv 2505.23458).
The `\typeset{...}` wrapper already gives the model a font condition, so this
needs no retraining. At each Euler step, contrast the human condition with
the font one instead of with the unconditional model:

```
v = v(label, font) + w · (v(label, human) − v(label, font))
```

Here `v(label, font)` is the velocity for `\typeset{label}` and
`v(label, human)` the velocity for the bare label. w = 1 is plain sampling,
and w > 1 moves further from font-like ink. Text guidance still needs its own
term, so a step costs three passes; [train_reflow.py](../scripts/train_reflow.py)
can distill the result into a few unguided steps.

- For: font data could then teach structure while guidance removes its look.
  This matters most for content no human wrote, such as truth-table rows or
  diagram labels: the font supplies the layout, guidance makes it look
  handwritten.
- Against: the guided direction contains everything that separates fonts
  from people (wobble, uneven spacing, baseline drift), so a large w can
  exaggerate it into ink messier than any human's. CFGRL also sees results
  fall past some w. Renders with hand noise would weaken the contrast.

Sweep w at 1, 1.5, 2 and 3 on human `valid` text. Measure human-likeness, for
example with a small classifier that tells real from generated ink, next to
legibility, which must not drop.

### Placing suggested items on the page

The suggestion model needs a way to say where a suggestion goes, so that a
text, box or arrow can be drawn in the right place. Both options below keep the
ink decoder free of page position. It draws each item in the item's own frame
(text at unit height, as the flow student already does; shapes from their
bbox or control points), and an affine transform puts it on the page.

#### Option A: one model predicts items with references, resolved deterministically

Status: `planned`

The suggestion model outputs layout items in the `layout.json` vocabulary, with
positions given relative to ink already on the page wherever possible:

| Target | Model output | Resolver |
|---|---|---|
| `node_outline` | `BOX rect AROUND fc_1.n3.text` | text bbox + padding |
| arrow / edge | `ARROW FROM n2.east TO n3.west` | node ports, then the orthogonal router in [diagram_layout.py](../scripts/diagram_layout.py) |
| `branch_label` | `TEXT "no" AT_EDGE e5` | label slot beside the edge |
| truth table cell | `CELL table_1 r=5 c=0 "1"` | grid from existing rules and cells |
| `table_rules` | `RULE table_1 after_row=3` | y from the row pitch |

Where nothing anchors the item, it falls back to quantized coordinate tokens
(Pix2Seq style, ~128–256 bins), normalized to a local frame such as the
element's bbox or the writer's text height. The probabilities of those tokens
feed into the suggestion's confidence.

- For: references are discrete and checkable. An unknown anchor, a failed
  route, or an ink collision ([validate_page.py](../scripts/validate_page.py))
  suppresses the suggestion. Labels can be derived from existing layouts,
  since items already store `bbox`, `points`, `from`/`to` and `row`/`col`.
- Against: the resolver assumes clean geometry. On real pages, with drifting
  rows and uneven spacing, it can place items visibly wrong.

#### Option B: a separate prediction model and placement model

Status: `planned`

A prediction model decides what to suggest. A separate placement model takes
the page context and the predicted tokens and predicts their position and
scale.

The prediction model should still output references for diagrams (which nodes
an arrow connects, which cell is next). For arrows the endpoints are the
suggestion, not a detail of placement. The placement model should predict a
correction to the deterministic estimate from Option A, not absolute
coordinates. That way it is never worse than the resolver where the geometry is
exact.

```
prediction model → (kind, content, references)
resolver         → initial geometry from references
placement model  → correction + per-coordinate uncertainty, from context ink
decoder          → ink in the item's frame, then transformed into place
```

- For: it can follow how a real writer actually spaces things. It trains on
  any page with known positions, with no structure labels needed. Either
  model can be replaced on its own. It is small, so it adds little latency.
- Against: the confidence gate depends on two models, so the placement model
  needs a calibrated uncertainty too. A correct suggestion in the wrong place
  is still a visibly wrong suggestion. Trained only on synthetic pages, it
  just copies [layout_engine.py](../scripts/layout_engine.py), so its benefit
  only shows on real handwriting, which is where it must be evaluated.

Plan: build Option A first as the baseline and fallback. Add Option B once
real pages show where the resolver's placement is off. Measure placement error
on real handwriting, and the false-suggestion rate with and without placement
in the gate.
