"""Convert MathWriting (data/mathwriting-2024) into student corpus shards, in
the format of scripts/generate_teacher_corpus.py, so train_student.py can
train on real handwritten math.

Only human-written inks are used: train/ becomes the training shards and
valid/ becomes the last shard, which Corpus.split holds out (keep
--val-shards 1). The text is the normalizedLabel; inks whose label is longer
than MAX_TEXT or uses characters outside the train charset are skipped.

MathWriting has no per-point character alignment, so every chars entry is -1
("unknown"). The student then learns only when a line is finished, not
which character each point draws.

Inks come from many devices, so each is normalized:
- scale: divided by the median stroke size (the larger side of each stroke's
  bounding box), roughly one symbol, so symbols are about the same size
  across inks whatever the screen resolution;
- sampling rate: every stroke is resampled to points --spacing symbol units
  apart along its path. Timestamps are dropped, as in the teacher corpus.

Offsets are (dx, dy, end_of_stroke) from the previous point, y down, the first
point at the origin, then normalized with the train set's mean and std.

Usage: python scripts/mathwriting_corpus.py [--source data/mathwriting-2024]
           [--out-dir data/mathwriting_corpus] [--lines-per-shard 8192]
           [--wandb-project P [--wandb-entity E] [--artifact mathwriting_corpus]]
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np

from corpus_artifact import add_wandb_args, push
from ink_corpus import MAX_TEXT

NS = "{http://www.w3.org/2003/InkML}"


def read_inkml(path):
    """(normalizedLabel, [stroke (n, 2) array of x, y])."""
    root = ET.parse(path).getroot()
    label = next(a.text for a in root.iter(NS + "annotation") if a.get("type") == "normalizedLabel")
    strokes = []
    for trace in root.iter(NS + "trace"):
        pts = [p.split() for p in trace.text.strip().split(",")]
        strokes.append(np.array([[float(p[0]), float(p[1])] for p in pts]))
    return label or "", strokes


def resample(stroke, step):
    """Points every `step` along the stroke's path, keeping both ends."""
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 0])
    stroke, seg = stroke[keep], seg[seg > 0]
    if len(stroke) < 2:
        return stroke[:1]
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, dist[-1], int(np.ceil(dist[-1] / step)) + 1)
    return np.stack([np.interp(s, dist, stroke[:, 0]), np.interp(s, dist, stroke[:, 1])], axis=1)


def to_offsets(strokes, spacing):
    """Raw (dx, dy, end_of_stroke) in symbol units, or None for an unusable ink."""
    sizes = [np.ptp(s, axis=0).max() for s in strokes if len(s)]
    unit = float(np.median(sizes)) if sizes else 0.0
    if unit <= 0:
        return None
    points, eos = [], []
    for s in strokes:
        if len(s):
            r = resample(s / unit, spacing)
            points.append(r)
            eos.append(np.arange(len(r)) == len(r) - 1)
    xy = np.concatenate(points)
    xy -= xy[0]
    d = np.diff(xy, axis=0, prepend=xy[:1])
    return np.concatenate([d, np.concatenate(eos)[:, None]], axis=1).astype(np.float32)


def convert(args):
    path, spacing, max_points = args
    label, strokes = read_inkml(path)
    if not label or len(label) > MAX_TEXT:
        return None
    offsets = to_offsets(strokes, spacing)
    if offsets is None or len(offsets) > max_points:
        return None
    return label, offsets


def load_split(source, split, spacing, max_points, workers):
    paths = sorted(glob.glob(os.path.join(source, split, "*.inkml")))
    with Pool(workers) as pool:
        out = pool.map(convert, [(p, spacing, max_points) for p in paths], chunksize=256)
    kept = [o for o in out if o is not None]
    print(f"{split}: kept {len(kept)} of {len(paths)} inks", flush=True)
    return kept


def write_shard(path, lines, mu, std):
    offsets = [(o - mu) / std for _, o in lines]
    np.savez(path,
             offsets=np.concatenate(offsets).astype(np.float32),
             chars=np.full(sum(len(o) for o in offsets), -1, dtype=np.int16),
             lengths=np.array([len(o) for o in offsets], dtype=np.int32),
             texts=np.array([t for t, _ in lines]),
             bias=np.zeros(len(lines), dtype=np.float32),
             rejected=np.int32(0))


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to convert again")
    os.makedirs(args.out_dir, exist_ok=True)
    train = load_split(args.source, "train", args.spacing, args.max_points, args.workers)
    valid = load_split(args.source, "valid", args.spacing, args.max_points, args.workers)

    charset = sorted({ch for t, _ in train for ch in t})
    allowed = set(charset)
    valid = [v for v in valid if set(v[0]) <= allowed]

    xy = np.concatenate([o[:, :2] for _, o in train])
    mu = np.array([*xy.mean(axis=0), 0.0], dtype=np.float32)
    std = np.array([*xy.std(axis=0), 1.0], dtype=np.float32)
    lengths = np.array([len(o) for _, o in train])
    print(f"points per ink p50/p90/p99: {np.percentile(lengths, [50, 90, 99]).round().tolist()}", flush=True)

    rng = np.random.default_rng(0)
    train = [train[i] for i in rng.permutation(len(train))]
    n_train = (len(train) + args.lines_per_shard - 1) // args.lines_per_shard
    for k in range(n_train):
        write_shard(os.path.join(args.out_dir, f"shard_{k:05d}.npz"),
                    train[k * args.lines_per_shard:(k + 1) * args.lines_per_shard], mu, std)
    write_shard(os.path.join(args.out_dir, f"shard_{n_train:05d}.npz"), valid, mu, std)

    with open(os.path.join(args.out_dir, "lines.txt"), "w") as f:
        f.write("\n".join(t for t, _ in train) + "\n")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        # text_height: ink units (after denormalizing) from descender to cap
        # line, for drawing previews at a whiteboard scale. A symbol is about
        # one unit, so this is a rough guess.
        json.dump({"charset": charset, "mu": mu.tolist(), "std": std.tolist(), "source": "mathwriting-2024",
                   "spacing": args.spacing, "text_height": 1.5, "lines": "lines.txt"}, f, indent=1)
    print(f"wrote {n_train} train shards and 1 validation shard ({len(valid)} inks) to {args.out_dir}", flush=True)
    if args.wandb_project:
        push(args.out_dir, args.wandb_entity, args.wandb_project, args.artifact)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="data/mathwriting-2024")
    ap.add_argument("--out-dir", default="data/mathwriting_corpus")
    ap.add_argument("--spacing", type=float, default=0.1, help="resampling step, in symbol sizes")
    ap.add_argument("--max-points", type=int, default=2000, help="skip longer inks after resampling")
    ap.add_argument("--lines-per-shard", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    add_wandb_args(ap, artifact_default="mathwriting_corpus")
    main(ap.parse_args())
