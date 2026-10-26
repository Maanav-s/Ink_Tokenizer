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
| Flow student trained on inks resampled to other point counts | `2789a1b`, [train_flow.py](../scripts/train_flow.py) | done (`font_flow_stretch`) | sample grids clearly better by eye; length sweep not yet run, see below |
| Flow student writing B-spline control points instead of points | `ad6a417`, [spline_flow_model.py](../scripts/spline_flow_model.py), [spline_corpus.py](../scripts/spline_corpus.py) | done | worked very well; splines are now the default ink representation, see below |

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

### Training on other point counts

Status: `done` (`models/student/font_flow_stretch`, W&B `inkEncoder`
run `fa2a16cb`); measured by eye only

The flow is trained at each ink's true point count but sampled at the length
head's prediction, so it never learned to write with a count that is off.
[length_sweep.py](../scripts/length_sweep.py) on `legibility_reward_flow_v2`
(run `w4a4tf0a`, 500 human `valid` labels, recognizer `all_sources_v1`)
showed the count matters:

| Point count | Token error rate | Exact |
|---|---|---|
| true | 0.055 | 0.750 |
| true × 0.6 | 0.133 | 0.486 |
| true × 0.8 | 0.051 | 0.722 |
| true × 1.25 | 0.058 | 0.714 |
| true × 1.5 | 0.079 | 0.662 |
| length head argmax | 0.049 | 0.750 |
| length head draw | 0.100 | 0.714 |

The draws are within 25% of the true count only 66% of the time, and they
double the error rate, mostly through dropped tokens (0.045 per token vs 0.021
at the true count). `8b3de3e` switched sampling to the argmax.

Run: continue `font_flow` on `data/font_corpus` for 20k steps with
`--stretch-prob 0.5 --stretch-max 1.25 --accumulate 2` (half the inks
resampled to 0.8–1.25× their points, log-uniform). Validation losses are
unchanged from the legibility runs (val_flow 0.191 vs 0.193), as expected,
since validation uses unresampled inks.

Result: the sample grids look much better, judged by eye. This has not been
measured. Next: run
`length_sweep.py --student models/student/font_flow_stretch/checkpoint.pt
--recognizer models/recognizer/all_sources_v1/checkpoint.pt` with the
settings above and compare it with the table. It should show a flatter error
rate across scales and a smaller gap between the draw and the true count.

### Splines instead of points

Status: `done`; the new default representation for text-to-ink students

The flow student writes every point, about a tenth of a symbol apart, so a
line is hundreds of positions and the curve is only as smooth as the smooth
loss makes it. [spline_corpus.py](../scripts/spline_corpus.py) fits each stroke
with a clamped cubic B-spline (least squares, chord-length parameters), and
[spline_flow_model.py](../scripts/spline_flow_model.py) generates the control
points with the flow student's architecture unchanged, then decodes them to
dense ink. On 512 MathWriting inks the fit has 3.3 points per control point
(`--points-per-control 4`; short strokes keep at least two) and an RMS error
of 0.02 ink units, a fifth of the point spacing, while keeping every stroke.

Expected: shorter sequences (faster, longer lines fit in a batch), curves
smooth by construction, and fewer points for the length head to get wrong.
Risks: one wrong control point moves a whole stretch of curve, and a
control point is less tied to one token than a point is.

Run as planned: `spline_corpus.py` on `data/mathwriting_corpus` (or
`data/font_corpus` for token alignment), then `train_spline_flow.py
--accumulate 2` (the point flow runs' 32768 per batch, counted in control
points, so a step sees about 3x the ink) for as many steps as the point flow
student it is compared with. Measure the recognizer's token error rate and
exact rate on human `valid` samples next to that student's, and the sampling
time per line. The flow losses are not comparable. Variants:
`--points-per-control` 2 and 8.

Result (`models/student/spline_flow`, W&B `splineDecoder` run
`41721da5`): MathWriting, default architecture (d=384, 8 layers),
`--points-per-control 4`, 50k steps of 32768 control points x 2, about
6.5 h. Final train / val losses: flow 0.292 / 0.376, smooth 0.175 / 0.329,
length 2.83 / 3.41 (over control points, so not comparable with the point
student's). No token alignment on MathWriting, so `char_acc` is 0.

It worked very well, a clear improvement over the point flow student, judged
by its samples. Splines replace points as the project's default ink
representation: new text-to-ink work should start from the spline student,
and the point flow student is the baseline.

The run covered 171 epochs, and val ends well above train (flow 0.376 vs
0.292), so it repeats the data heavily. Font renders (`data/font_corpus`)
would add data and token alignment. Still to measure: the recognizer's token
error and exact rates against the point student.

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
   which weights the term by t³ (t drawn from [0.5, 1] by default, set by
   `--legibility-t-min`), in a second pass separate from the flow loss's
   times. Not yet run.
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

Status: `planned`; code ready: `FlowStudent.guide_from_fonts`, and
`--font-guidance` in [measure_legibility.py](../scripts/measure_legibility.py)
and [visualize_flow.py](../scripts/visualize_flow.py). First run: the spline
student continued on `font_corpus.py --splits train synthetic`, converted by
`spline_corpus.py`

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
