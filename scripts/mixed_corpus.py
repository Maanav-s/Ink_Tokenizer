r"""Merge the MathWriting corpus (scripts/mathwriting_corpus.py) and the
teacher corpus (scripts/generate_teacher_corpus.py) into one corpus, so that
one model writes both formulas and plain text.

A teacher line `some words` becomes `\text{some words}`. MathWriting has no
\text, so the wrapper is what tells a model to write in the teacher's hand
rather than as a formula. Teacher lines too long to fit MAX_TEXT once wrapped
are dropped.

The merged corpus keeps MathWriting's conventions, so its shards are used as
they are (symlinked). Teacher inks are converted to them:
- scale: TEACHER_TEXT_HEIGHT teacher units become MathWriting's text_height;
- sampling rate: every stroke is resampled to points `spacing` apart along its
  path, as MathWriting's are. A resampled point keeps the character label of
  the teacher point it replaces, shifted past the `\text{` tokens;
- normalization: MathWriting's mean and std.

The vocabulary is MathWriting's with the new tokens appended, so token ids of
a model trained on MathWriting stay valid (train_flow.py --init-from).

The last shard is the validation set: MathWriting's validation shard plus the
teacher's last shard. All other shards are for training.

Usage: python scripts/mixed_corpus.py [--mathwriting data/mathwriting_corpus]
           [--teacher data/teacher_corpus] [--out-dir data/mixed_corpus]
"""
import argparse
import glob
import json
import os

import numpy as np

from ink_corpus import MAX_TEXT
from sample_student import TEACHER_TEXT_HEIGHT

SHARD_KEYS = ("offsets", "chars", "lengths", "texts", "bias")
WRAPPER_TOKENS = 3  # \text, { and }
LABEL_SHIFT = 2     # tokens before the line's first character


def resample(stroke, labels, step):
    """A stroke (n, 2) as points `step` apart along its path, each with the
    label of the original point it replaces."""
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(stroke, axis=0), axis=1))])
    s = np.linspace(0.0, dist[-1], int(np.ceil(dist[-1] / step)) + 1)
    xy = np.stack([np.interp(s, dist, stroke[:, 0]), np.interp(s, dist, stroke[:, 1])], axis=1)
    return xy, labels[np.searchsorted(dist, s).clip(max=len(stroke) - 1)]


def convert_ink(ink, labels, scale, spacing):
    """Denormalized teacher (dx, dy, end_of_stroke) and per-point labels ->
    the same ink in MathWriting's units and sampling, still denormalized."""
    xy = np.cumsum(ink[:, :2], axis=0) * scale
    ends = np.flatnonzero(ink[:, 2] > 0.5) + 1
    points, out_labels, eos = [], [], []
    for stroke, stroke_labels in zip(np.split(xy, ends), np.split(labels, ends)):
        if len(stroke):
            p, l = resample(stroke, stroke_labels, spacing)
            points.append(p)
            out_labels.append(l)
            eos.append(np.arange(len(p)) == len(p) - 1)
    xy = np.concatenate(points)
    d = np.diff(xy, axis=0, prepend=xy[:1])
    return np.concatenate([d, np.concatenate(eos)[:, None]], axis=1).astype(np.float32), np.concatenate(out_labels)


def convert_shard(path, teacher_meta, meta, max_points):
    """A teacher shard as a dict of merged-corpus shard arrays, and how many
    of its lines were dropped."""
    d = np.load(path)
    teacher_mu, teacher_std = np.array(teacher_meta["mu"]), np.array(teacher_meta["std"])
    mu, std = np.array(meta["mu"]), np.array(meta["std"])
    scale = meta["text_height"] / TEACHER_TEXT_HEIGHT
    out = {k: [] for k in SHARD_KEYS}
    start, dropped = 0, 0
    for text, n, bias in zip(d["texts"], d["lengths"], d["bias"]):
        ink, labels = d["offsets"][start:start + n] * teacher_std + teacher_mu, d["chars"][start:start + n]
        start += n
        if len(text) + WRAPPER_TOKENS > MAX_TEXT:
            dropped += 1
            continue
        offsets, labels = convert_ink(ink, labels, scale, meta["spacing"])
        if len(offsets) > max_points:
            dropped += 1
            continue
        out["offsets"].append((offsets - mu) / std)
        out["chars"].append(labels + LABEL_SHIFT)
        out["lengths"].append(len(offsets))
        out["texts"].append("\\text{" + str(text) + "}")
        out["bias"].append(bias)
    return {"offsets": np.concatenate(out["offsets"]).astype(np.float32),
            "chars": np.concatenate(out["chars"]).astype(np.int16),
            "lengths": np.array(out["lengths"], dtype=np.int32),
            "texts": np.array(out["texts"]),
            "bias": np.array(out["bias"], dtype=np.float32)}, dropped


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to merge again")
    math_shards = sorted(glob.glob(os.path.join(args.mathwriting, "shard_*.npz")))
    teacher_shards = sorted(glob.glob(os.path.join(args.teacher, "shard_*.npz")))
    if len(math_shards) < 2 or len(teacher_shards) < 2:
        raise SystemExit(f"need at least 2 shards in each of {args.mathwriting} and {args.teacher}, "
                         f"found {len(math_shards)} and {len(teacher_shards)}")
    meta = json.load(open(os.path.join(args.mathwriting, "meta.json")))
    teacher_meta = json.load(open(os.path.join(args.teacher, "meta.json")))
    os.makedirs(args.out_dir, exist_ok=True)

    def shard_path(index):
        return os.path.join(args.out_dir, f"shard_{index:05d}.npz")

    count = 0
    for path in math_shards[:-1]:
        os.symlink(os.path.relpath(path, args.out_dir), shard_path(count))
        count += 1
    texts, kept, dropped = [], 0, 0
    for path in teacher_shards[:-1]:
        shard, n = convert_shard(path, teacher_meta, meta, args.max_points)
        np.savez(shard_path(count), **shard, rejected=np.int32(0))
        count += 1
        texts.extend(shard["texts"])
        kept += len(shard["texts"])
        dropped += n
    print(f"teacher: kept {kept} training lines, dropped {dropped} (too long once wrapped, "
          f"or over {args.max_points} points)", flush=True)

    held_out_math = np.load(math_shards[-1])
    held_out_teacher, _ = convert_shard(teacher_shards[-1], teacher_meta, meta, args.max_points)
    np.savez(shard_path(count), rejected=np.int32(0),
             **{k: np.concatenate([held_out_math[k], held_out_teacher[k]]) for k in SHARD_KEYS})
    print(f"validation: {len(held_out_math['texts'])} MathWriting inks and "
          f"{len(held_out_teacher['texts'])} teacher lines", flush=True)

    math_lines = open(os.path.join(args.mathwriting, meta["lines"])).read()
    with open(os.path.join(args.out_dir, "lines.txt"), "w") as f:
        f.write(math_lines + "\n".join(texts) + "\n")
    new_tokens = [t for t in ["\\text", *teacher_meta["charset"]] if t not in meta["charset"]]
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({**meta, "charset": meta["charset"] + new_tokens,
                   "source": [meta["source"], teacher_meta["teacher"]], "lines": "lines.txt"}, f, indent=1)
    print(f"{len(meta['charset'])} MathWriting tokens + {new_tokens}", flush=True)
    print(f"wrote {count} train shards and 1 validation shard to {args.out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mathwriting", default="data/mathwriting_corpus")
    ap.add_argument("--teacher", default="data/teacher_corpus")
    ap.add_argument("--out-dir", default="data/mixed_corpus")
    ap.add_argument("--max-points", type=int, default=2000, help="skip longer inks after resampling")
    main(ap.parse_args())
