"""Shared pieces of the converters for human handwritten text lines
(scripts/iamondo_corpus.py, deepwriting_corpus.py, brush_corpus.py): ink
normalization that keeps a label on every point, and writing a corpus in the
format of scripts/mathwriting_corpus.py.

Inks are normalized as MathWriting's are, so the corpora can be pooled with it
(scripts/pool_corpus.py): scaled by the median stroke size and resampled every
`spacing` units along each stroke. A resampled point takes the label of the
nearest original point.
"""
import json
import os

import numpy as np

TEXT_HEIGHT = 1.5  # as mathwriting_corpus: a rough guess, a symbol being about one unit


def resample_labeled(stroke, labels, step):
    """mathwriting_corpus.resample, carrying each point's label along."""
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 0])
    stroke, labels, seg = stroke[keep], labels[keep], seg[seg > 0]
    if len(stroke) < 2:
        return stroke[:1], labels[:1]
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, dist[-1], int(np.ceil(dist[-1] / step)) + 1)
    xy = np.stack([np.interp(s, dist, stroke[:, 0]), np.interp(s, dist, stroke[:, 1])], axis=1)
    i = np.searchsorted(dist, s).clip(1, len(dist) - 1)
    i -= (s - dist[i - 1]) < (dist[i] - s)
    return xy, labels[i]


def ink_offsets(strokes, labels, spacing):
    """Strokes [(n, 2)] in writing order, y down, with per-point labels [(n,)]
    -> raw (dx, dy, end_of_stroke) in symbol units, the resampled labels, the
    first point's position and the input units per symbol unit; None for an
    ink with no extent."""
    sizes = [np.ptp(s, axis=0).max() for s in strokes if len(s)]
    unit = float(np.median(sizes)) if sizes else 0.0
    if unit <= 0:
        return None
    points, eos, out_labels = [], [], []
    for s, l in zip(strokes, labels):
        if len(s):
            r, rl = resample_labeled(s / unit, l, spacing)
            points.append(r)
            out_labels.append(rl)
            eos.append(np.arange(len(r)) == len(r) - 1)
    xy = np.concatenate(points)
    origin = xy[0] * unit
    xy -= xy[0]
    d = np.diff(xy, axis=0, prepend=xy[:1])
    offsets = np.concatenate([d, np.concatenate(eos)[:, None]], axis=1).astype(np.float32)
    return offsets, np.concatenate(out_labels), origin, unit


def split_strokes(xy, ends):
    """Points (n, 2) and a per-point end-of-stroke flag -> one array per
    stroke; also returns the split indices, to split labels the same way."""
    cuts = np.flatnonzero(ends[:-1]) + 1
    return np.split(xy, cuts), cuts


def held_out_writers(writers, every=20):
    """Every `every`-th writer (sorted), so validation has unseen writers."""
    return set(sorted(writers)[::every])


def write_shard(path, lines, mu, std):
    offsets = [(l["offsets"] - mu) / std for l in lines]
    np.savez(path,
             offsets=np.concatenate(offsets).astype(np.float32),
             chars=np.concatenate([l["chars"] for l in lines]).astype(np.int16),
             lengths=np.array([len(o) for o in offsets], dtype=np.int32),
             texts=np.array([l["text"] for l in lines]),
             bias=np.zeros(len(lines), dtype=np.float32),
             rejected=np.int32(0),
             writers=np.array([l["writer"] for l in lines]))


def write_corpus(out_dir, train, valid, source, spacing, lines_per_shard):
    """Train lines in shuffled shards and the validation lines in the last
    one, each line a dict of text, raw offsets, chars (index into the text, -1
    unknown) and writer. The charset and normalization come from the train
    lines; validation lines with other characters are dropped. Returns how
    many were."""
    charset = sorted({ch for l in train for ch in l["text"]})
    allowed = set(charset)
    kept = [l for l in valid if set(l["text"]) <= allowed]

    xy = np.concatenate([l["offsets"][:, :2] for l in train])
    mu = np.array([*xy.mean(axis=0), 0.0], dtype=np.float32)
    std = np.array([*xy.std(axis=0), 1.0], dtype=np.float32)
    lengths = np.array([len(l["offsets"]) for l in train])
    print(f"points per line p50/p90/p99: {np.percentile(lengths, [50, 90, 99]).round().tolist()}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    train = [train[i] for i in rng.permutation(len(train))]
    n_train = (len(train) + lines_per_shard - 1) // lines_per_shard
    for k in range(n_train):
        write_shard(os.path.join(out_dir, f"shard_{k:05d}.npz"),
                    train[k * lines_per_shard:(k + 1) * lines_per_shard], mu, std)
    write_shard(os.path.join(out_dir, f"shard_{n_train:05d}.npz"), kept, mu, std)
    with open(os.path.join(out_dir, "lines.txt"), "w") as f:
        f.write("\n".join(l["text"] for l in train) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"charset": charset, "tokenizer": "char", "mu": mu.tolist(), "std": std.tolist(),
                   "source": source, "spacing": spacing, "text_height": TEXT_HEIGHT, "lines": "lines.txt"},
                  f, indent=1)
    print(f"{len(charset)} tokens; wrote {n_train} train shards ({len(train)} lines) and 1 validation shard "
          f"({len(kept)} lines, {len({l['writer'] for l in kept})} writers) to {out_dir}", flush=True)
    return len(valid) - len(kept)
