# Text-to-ink conditioning experiments

These are two fresh-training variants of the same text-to-ink student. Both use
six Mamba2 layers at width 256, four heads of residual cross-attention, and the
existing mixture-density / character-index heads and losses. The tokenizer,
ink representation, optimizer, and default 50,000-step schedule are unchanged.
No new dependencies are required. Train on a cluster.

## Unpadded text prefix

`--conditioning unpadded` uses `[text tokens, SEP, ink]`. Before running Mamba,
the model groups rows by actual token count and strips all leading PAD tokens.
Each group has a separate recurrent history and inference cache. Results are
restored to the input row order, and sampling's per-line bias follows each row.
This removes unwanted prefix state updates rather than zeroing embeddings.
Cross-attention still reads the positioned text tokens and SEP.

Both new variants use the same training and validation batches with equal text
token counts, bucketed by ink length within each count. This avoids dozens of
separate Mamba calls per batch in the unpadded variant and keeps batch composition
consistent across the comparison.
Mixed-length previews and external calls are supported by grouping internally.
Grouping changes batch composition relative to legacy runs and may reduce
throughput; compare held-out samples as well as losses and elapsed time.
Sampled coordinates are stochastic:
changing batch companions can change which random draws a line receives even
though its conditional hidden states no longer depend on PAD count.

## Transformer prefix

`--conditioning transformer` uses a two-layer bidirectional transformer text
encoder with four heads, width 256, a GELU feedforward width of 1024, and no
dropout. Positioned text tokens and SEP are encoded with the ordinary attention
padding mask. A transformer decoder layer uses 32 learned queries to attend to
that encoded text and produces 32 contextual slots. Query slots also attend to
one another. Mamba receives `[32 slots, SEP, ink]`; its final residual
cross-attention reads the slots. The text encoder, slots, and attention memory
are computed once during sampling and trained jointly with the ink decoder.

The encoder is learned from scratch; it has no explicit LaTeX grammar or parsing
supervision. Contextual attention can learn token relationships, but that does
not guarantee correct parsing. Thirty-two slots may lose fine detail in longer
expressions; `--prefix-slots 50` is an available comparison. Encoder depth and
head count are configurable with `--text-encoder-layers` and
`--text-encoder-heads` (head counts must divide model width).

## Cluster commands

Use new run directories. The helper chooses a distinct directory for each
architecture/corpus pair. Run these jobs separately or on different GPUs; set
`INK_TRAIN_GPU` for a GPU other than 0. On the lab cluster, invoke directly:

```bash
scripts/train_conditioning.sh unpadded teacher \
  --wandb-entity maanavsikaria-university-of-texas-at-austin --wandb-project inkTokensMamba
scripts/train_conditioning.sh transformer teacher \
  --wandb-entity maanavsikaria-university-of-texas-at-austin --wandb-project inkTokensMamba

scripts/train_conditioning.sh unpadded mathwriting \
  --wandb-entity maanavsikaria-university-of-texas-at-austin --wandb-project inkTokensMamba
scripts/train_conditioning.sh transformer mathwriting \
  --wandb-entity maanavsikaria-university-of-texas-at-austin --wandb-project inkTokensMamba
```

Teacher jobs default to `teacher_corpus:latest`; pass an explicit artifact
version to train both on the same immutable data. MathWriting jobs use the local
`data/mathwriting_corpus` and log metrics without pulling an artifact. If that
corpus is absent, prepare it with the existing `scripts/fetch_mathwriting.sh`
on the cluster. For simultaneous jobs, sync the environment once with
`scripts/run_in_apptainer.sh cpu true`, then use `INK_NO_SYNC=1`.

Extra arguments override helper defaults, e.g.:

```bash
scripts/train_conditioning.sh transformer mathwriting --prefix-slots 50 \
  --run-dir models/student/mathwriting_transformer_50
```

On TACC, wrap a command using `scripts/submit_slurm.sh` with your partition and
`--module tacc-apptainer/1.4.1 --`; see [slurm.md](slurm.md).

## Checkpoints and evaluation

The checkpoint records conditioning type and transformer dimensions. Resuming
an existing run directory always keeps its saved architecture, regardless of
new CLI flags. Old checkpoints without a conditioning field use `legacy`,
retaining the original left-padded behavior so they load and resume faithfully.
Do not use an old run directory to start either experiment.

Sampling automatically selects the checkpoint architecture:

```bash
scripts/run_in_apptainer.sh 0 python scripts/sample_student.py \
  --checkpoint models/student/teacher_unpadded/checkpoint.pt \
  "pine" "recognizable refrigerated direr moleskin plantain." --out artifacts/teacher_unpadded.png
scripts/run_in_apptainer.sh 0 python scripts/sample_student.py \
  --checkpoint models/student/mathwriting_transformer/checkpoint.pt \
  'F' '\frac{x_1}{x_2}' --out artifacts/mathwriting_transformer.png
```

Inspect held-out generation for legibility, spelling, layout, stroke boundaries,
and termination. MathWriting `char_acc` still measures stopping, because it has
no per-point alignment; it cannot measure correct symbols. Neither variant
changes pen-lift supervision or the MDN, so the previously observed low
MathWriting pen-lift recall may persist. The preview alternates reference and
student rows and clips extents; inspect individual generated lines as well.
