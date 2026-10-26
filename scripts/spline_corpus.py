"""Convert a corpus of point inks into one of B-spline control points, for the
spline flow student (scripts/spline_flow_model.py).

Each stroke is fit, by least squares, with a clamped uniform B-spline of
degree up to 3, parameterized by chord length. A stroke of k points gets
m = 1 + ceil((k - 1) / --points-per-control) control points (at least 2 if
k >= 2, and 1 for a dot), so sequences are about that many times shorter.
Decoding evaluates each stroke's spline at (m - 1) * points-per-control + 1
evenly spaced parameters, which is close to the original count.

The shards keep the source's format and normalization (meta.json mu and
std), with control points in place of points: offsets are between
consecutive control points, and the pen flag marks a stroke's last control
point. A control point takes the token label of the original point nearest
its parameter, so font inks keep their alignment. meta.json gains "spline":
the source directory and points_per_control. The source's validation shard
stays last.

Usage: python scripts/spline_corpus.py [--source data/mathwriting_corpus]
           [--out-dir data/spline_corpus] [--points-per-control 4]
"""
import argparse
import glob
import json
import math
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

DEGREE = 3


def basis(u, m):
    """Clamped uniform B-spline basis of m control points at parameters u in
    [0, 1]: (len(u), m). The degree is DEGREE, or m - 1 if that is lower."""
    degree = min(DEGREE, m - 1)
    knots = np.concatenate([np.zeros(degree), np.linspace(0, 1, m - degree + 1), np.ones(degree)])
    u = u[:, None]
    N = ((knots[:-1] <= u) & (u < knots[1:])).astype(float)
    # u = 1 is outside every half-open span; it belongs to the last one.
    N[u[:, 0] >= 1, :] = 0
    N[u[:, 0] >= 1, m - 1] = 1
    for d in range(1, degree + 1):
        left = knots[d:-1] - knots[:-d - 1]
        right = knots[d + 1:] - knots[1:-d]
        a = np.divide(u - knots[:-d - 1], left, out=np.zeros((len(u), len(left))), where=left > 0)
        b = np.divide(knots[d + 1:] - u, right, out=np.zeros((len(u), len(right))), where=right > 0)
        N = a * N[:, :-1] + b * N[:, 1:]
    return N


def stroke_bounds(pen):
    """(start, size) of each stroke, from an ink's pen flags; the last point
    always ends a stroke."""
    ends = np.flatnonzero(pen > 0.5)
    if not len(ends) or ends[-1] != len(pen) - 1:
        ends = np.append(ends, len(pen) - 1)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends - starts + 1


def to_offsets(xy, sizes, mu, std):
    """Absolute positions of consecutive strokes -> normalized offsets."""
    raw = np.diff(xy, axis=0, prepend=np.zeros((1, 2)))
    pen = np.zeros(len(xy))
    pen[np.cumsum(sizes) - 1] = 1
    return np.concatenate([(raw - mu) / std, pen[:, None]], axis=1)


def fit_ink(offsets, chars, points_per_control, mu, std):
    """One ink's normalized offsets (k, 3) and token labels (k,) -> its
    control points as normalized offsets (n, 3), their labels (n,), and the
    sum of squared fit errors over the ink's points (in ink units)."""
    xy = np.cumsum(offsets[:, :2] * std + mu, axis=0)
    control, labels, sizes, error = [], [], [], 0.0
    for s, k in zip(*stroke_bounds(offsets[:, 2])):
        points = xy[s:s + k]
        m = 1 if k == 1 else min(k, 1 + math.ceil((k - 1) / points_per_control))
        chord = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        u = chord / chord[-1] if chord[-1] > 0 else np.linspace(0, 1, k)
        A = basis(u, m)
        c = np.linalg.lstsq(A, points, rcond=None)[0]
        error += float(((A @ c - points) ** 2).sum())
        nearest = np.abs(u[None, :] - np.linspace(0, 1, m)[:, None]).argmin(axis=1)
        control.append(c)
        labels.append(chars[s:s + k][nearest])
        sizes.append(m)
    return (to_offsets(np.concatenate(control), sizes, mu, std).astype(offsets.dtype),
            np.concatenate(labels).astype(chars.dtype), error)


def decode_ink(control, chars, points_per_control, mu, std):
    """The inverse of fit_ink: control points as normalized offsets (n, 3) and
    labels (n,) -> dense normalized offsets and labels. Each point takes the
    label of the control point nearest its parameter."""
    xy = np.cumsum(control[:, :2] * std + mu, axis=0)
    points, labels, sizes = [], [], []
    for s, m in zip(*stroke_bounds(control[:, 2])):
        n = 1 if m == 1 else (m - 1) * points_per_control + 1
        u = np.linspace(0, 1, n)
        points.append(basis(u, m) @ xy[s:s + m])
        labels.append(chars[s:s + m][np.rint(u * (m - 1)).astype(int)])
        sizes.append(n)
    return (to_offsets(np.concatenate(points), sizes, mu, std).astype(control.dtype),
            np.concatenate(labels).astype(chars.dtype))


def decode_weights(pen, points_per_control):
    """For control points with these pen flags, decode_ink's points as
    weighted sums of control points: per point, the indices (n, DEGREE + 1)
    of the control points it mixes and their weights (zero where a stroke
    has fewer), and its pen flag (n,)."""
    index, weight, ends = [], [], []
    for s, m in zip(*stroke_bounds(pen)):
        n = 1 if m == 1 else (m - 1) * points_per_control + 1
        # A point of a B-spline mixes at most DEGREE + 1 control points.
        B = np.pad(basis(np.linspace(0, 1, n), m), ((0, 0), (0, max(0, DEGREE + 1 - m))))
        cols = np.argsort(-B, axis=1, kind="stable")[:, :DEGREE + 1]
        weight.append(np.take_along_axis(B, cols, axis=1))
        index.append(s + np.minimum(cols, m - 1))
        ends.append(np.arange(n) == n - 1)
    return np.concatenate(index), np.concatenate(weight), np.concatenate(ends)


def convert_shard(path, out_dir, points_per_control, mu, std):
    d = np.load(path)
    starts = np.concatenate([[0], np.cumsum(d["lengths"])[:-1]])
    offsets, chars, lengths, error = [], [], [], 0.0
    for s, k in zip(starts, d["lengths"]):
        off, ch, e = fit_ink(d["offsets"][s:s + k], d["chars"][s:s + k], points_per_control, mu, std)
        offsets.append(off)
        chars.append(ch)
        lengths.append(len(off))
        error += e
    out = {k: d[k] for k in d.files}
    out.update(offsets=np.concatenate(offsets), chars=np.concatenate(chars),
               lengths=np.array(lengths, dtype=d["lengths"].dtype))
    np.savez(os.path.join(out_dir, os.path.basename(path)), **out)
    return int(d["lengths"].sum()), int(sum(lengths)), error


def main(args):
    meta = json.load(open(os.path.join(args.source, "meta.json")))
    shards = sorted(glob.glob(os.path.join(args.source, "shard_*.npz")))
    if not shards:
        raise ValueError(f"no shards in {args.source}")
    os.makedirs(args.out_dir, exist_ok=True)
    mu, std = np.array(meta["mu"][:2]), np.array(meta["std"][:2])
    convert = partial(convert_shard, out_dir=args.out_dir, points_per_control=args.points_per_control, mu=mu, std=std)
    points = controls = 0
    error = 0.0
    with Pool(args.workers) as pool:
        for path, (p, c, e) in zip(shards, pool.imap(convert, shards)):
            points, controls, error = points + p, controls + c, error + e
            print(f"{os.path.basename(path)}: {p} points -> {c} control points", flush=True)
    meta["spline"] = {"source": args.source, "points_per_control": args.points_per_control}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)
    spacing = meta.get("spacing")
    print(f"{points / controls:.2f} points per control point; fit RMS error {math.sqrt(error / points):.4f} ink units"
          + (f" (points are {spacing} apart)" if spacing else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="data/mathwriting_corpus")
    ap.add_argument("--out-dir", default="data/spline_corpus")
    ap.add_argument("--points-per-control", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
