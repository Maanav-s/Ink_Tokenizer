r"""Render MathWriting's LaTeX labels in single-stroke fonts and add them to a
corpus as extra training inks, so a model sees the exact layout and tokens of
every expression it learns to write.

Every unique label of --splits (default: train) is typeset once per font in
--fonts (scripts/math_typeset.py, scripts/fonts.py), as clean strokes in
writing order. Labels the typesetter does not support are skipped.

A rendered label is stored as `\typeset{label}`. MathWriting has no \typeset,
so the wrapper marks font ink, as mixed_corpus.py's \text marks the teacher's
hand: a model trained on both writes in the human hand when it is left out.

Unlike MathWriting's, rendered inks are aligned: every point is labelled with
the index of the token it draws (a fraction bar with \frac, a delimiter with
its own token), counted from the start of `\typeset{`.

Inks are converted as scripts/mathwriting_corpus.py converts MathWriting's (y
down, scaled by the median stroke size, resampled every `spacing`) and
normalized with the base corpus's mean and std, so a checkpoint trained on the
base corpus can continue on this one (train_flow.py --init-from).

The base corpus's shards are used as they are (symlinked): its training
shards, then the rendered shards, then its validation shard. That one stays
last, so validation is on human ink only. The vocabulary is the base's with
\typeset appended.

MathWriting's `valid` labels are rendered too, into font_valid.npz rather than
a shard, so that only the recognizer (scripts/train_recognizer.py) validates
on them.

Usage: python scripts/font_corpus.py [--base data/mathwriting_corpus]
           [--source data/mathwriting-2024] [--splits train]
           [--fonts EMSReadability EMSTech] [--out-dir data/font_corpus]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

import fonts
import math_typeset
from ink_corpus import LATEX_TOKEN, MAX_TEXT, split_text
from mathwriting_corpus import read_inkml
from mixed_corpus import SHARD_KEYS, resample

WRAPPER = "\\typeset"
LABEL_SHIFT = 2  # \typeset and {
VALID_FILE = "font_valid.npz"  # not shard_*, so flow students' validation stays human ink

_renderers = {}


def render(label, font):
    """The label typeset in `font`: strokes ((n, 2) arrays, y down) and the
    index of the label token each point draws, or None if it cannot be typeset."""
    if font not in _renderers:
        _renderers[font] = fonts.TextRenderer(font)
    renderer = _renderers[font]
    token_at = np.zeros(len(label), dtype=np.int64)
    for k, m in enumerate(LATEX_TOKEN.finditer(label)):
        token_at[m.start():m.end()] = k
    try:
        eq = math_typeset.typeset(label, 1.0, renderer)
        strokes, labels = [], []
        for atom in eq.atoms:
            if atom["type"] == "char":
                parts = [[(atom["x"] + x, atom["y"] + y) for x, y in st]
                         for st in renderer.char_strokes(atom["ch"], atom["size"])]
            else:
                parts = atom["strokes"]
            for st in parts:
                strokes.append(np.array(st) * [1.0, -1.0])
                labels.append(np.full(len(st), token_at[atom["src"]]))
    except ValueError:
        return None
    return strokes, labels


def to_offsets(strokes, labels, spacing):
    """(dx, dy, end_of_stroke) in symbol units, as mathwriting_corpus.to_offsets
    makes them, and each point's label."""
    unit = float(np.median([np.ptp(s, axis=0).max() for s in strokes]))
    points, out_labels, eos = [], [], []
    for stroke, stroke_labels in zip(strokes, labels):
        keep = np.concatenate([[True], np.linalg.norm(np.diff(stroke, axis=0), axis=1) > 0])
        p, l = resample(stroke[keep] / unit, stroke_labels[keep], spacing)
        points.append(p)
        out_labels.append(l)
        eos.append(np.arange(len(p)) == len(p) - 1)
    xy = np.concatenate(points)
    xy -= xy[0]
    d = np.diff(xy, axis=0, prepend=xy[:1])
    return np.concatenate([d, np.concatenate(eos)[:, None]], axis=1).astype(np.float32), np.concatenate(out_labels)


def convert(job):
    label, font, spacing, max_points = job
    rendered = render(label, font)
    if rendered is None:
        return "unsupported"
    offsets, labels = to_offsets(*rendered, spacing)
    if len(offsets) > max_points:
        return "too long"
    return WRAPPER + "{" + label + "}", offsets, labels + LABEL_SHIFT


def read_label(path):
    return read_inkml(path)[0]


def write_shard(path, lines, mu, std):
    np.savez(path,
             offsets=np.concatenate([(o - mu) / std for _, o, _ in lines]).astype(np.float32),
             chars=np.concatenate([c for _, _, c in lines]).astype(np.int16),
             lengths=np.array([len(o) for _, o, _ in lines], dtype=np.int32),
             texts=np.array([t for t, _, _ in lines]),
             bias=np.zeros(len(lines), dtype=np.float32),
             rejected=np.int32(0))


def render_splits(splits, args, meta):
    """Every unique label of the MathWriting splits that the base vocabulary
    covers, rendered in each of args.fonts, as (text, offsets, labels)."""
    paths = sorted(p for split in splits for p in glob.glob(os.path.join(args.source, split, "*.inkml")))
    with Pool(args.workers) as pool:
        labels = sorted(set(pool.map(read_label, paths, chunksize=256)))
    allowed = set(meta["charset"])
    labels = [t for t in labels if t and len(split_text(t, "latex")) + 3 <= MAX_TEXT
              and set(split_text(t, "latex")) <= allowed]
    if args.max_labels:
        labels = labels[:args.max_labels]
    print(f"{len(labels)} unique labels from {len(paths)} inks in {splits}", flush=True)

    jobs = [(t, font, meta["spacing"], args.max_points) for t in labels for font in args.fonts]
    with Pool(args.workers) as pool:
        out = pool.map(convert, jobs, chunksize=64)
    kept = [o for o in out if not isinstance(o, str)]
    print(f"rendered {len(kept)} of {len(jobs)} (label, font) pairs; "
          f"{out.count('unsupported')} unsupported by the typesetter, {out.count('too long')} over "
          f"{args.max_points} points", flush=True)
    return kept


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to build it again")
    if {"valid", "test"} & set(args.splits):
        raise SystemExit("rendering valid or test labels would leak held-out expressions into training")
    base_shards = sorted(glob.glob(os.path.join(args.base, "shard_*.npz")))
    if len(base_shards) < 2:
        raise SystemExit(f"need at least 2 shards in {args.base}, found {len(base_shards)}")
    meta = json.load(open(os.path.join(args.base, "meta.json")))
    if meta.get("tokenizer") != "latex":
        raise SystemExit(f"{args.base} does not use the latex tokenizer")
    for font in args.fonts:
        fonts.TextRenderer(font)  # fail now on a misspelled font, not in every worker

    kept = render_splits(args.splits, args, meta)
    lengths = np.array([len(o) for _, o, _ in kept])
    print(f"points per ink p50/p90/p99: {np.percentile(lengths, [50, 90, 99]).round().tolist()}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    def shard_path(index):
        return os.path.join(args.out_dir, f"shard_{index:05d}.npz")

    count = 0
    for path in base_shards[:-1]:
        os.symlink(os.path.relpath(path, args.out_dir), shard_path(count))
        count += 1
    rng = np.random.default_rng(0)
    kept = [kept[i] for i in rng.permutation(len(kept))]
    mu, std = np.array(meta["mu"], dtype=np.float32), np.array(meta["std"], dtype=np.float32)
    for k in range(0, len(kept), args.lines_per_shard):
        write_shard(shard_path(count), kept[k:k + args.lines_per_shard], mu, std)
        count += 1
    os.symlink(os.path.relpath(base_shards[-1], args.out_dir), shard_path(count))

    held_out = render_splits(["valid"], args, meta)
    write_shard(os.path.join(args.out_dir, VALID_FILE), held_out, mu, std)
    print(f"wrote {len(held_out)} rendered valid labels to {VALID_FILE}", flush=True)

    base_lines = open(os.path.join(args.base, meta["lines"])).read()
    with open(os.path.join(args.out_dir, "lines.txt"), "w") as f:
        f.write(base_lines + "\n".join(t for t, _, _ in kept) + "\n")
    new_tokens = [t for t in (WRAPPER, "{", "}") if t not in meta["charset"]]
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({**meta, "charset": meta["charset"] + new_tokens,
                   "source": [meta["source"], {"fonts": args.fonts, "splits": args.splits}],
                   "lines": "lines.txt"}, f, indent=1)
    print(f"appended tokens {new_tokens}", flush=True)
    print(f"wrote {count} train shards ({len(base_shards) - 1} from {args.base}) and 1 validation shard "
          f"to {args.out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="data/mathwriting_corpus",
                    help="corpus whose shards and ink units are kept (mathwriting_corpus or mixed_corpus)")
    ap.add_argument("--source", default="data/mathwriting-2024")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="MathWriting splits whose labels are rendered; never valid or test")
    ap.add_argument("--fonts", nargs="+", default=["EMSReadability", "EMSTech"])
    ap.add_argument("--out-dir", default="data/font_corpus")
    ap.add_argument("--max-points", type=int, default=2000, help="skip longer inks after resampling")
    ap.add_argument("--max-labels", type=int, help="render only the first N labels, for a quick test")
    ap.add_argument("--lines-per-shard", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
