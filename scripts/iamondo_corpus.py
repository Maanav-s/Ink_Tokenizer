r"""Convert IAMonDo (data/IAMonDo-db-1.0, real handwritten documents) into
corpus shards in the format of scripts/mathwriting_corpus.py, with each point
aligned to the word it belongs to.

read_document() parses one document into strokes (x, y in mm on the page, t
in seconds) and its annotation tree: Document > Textblock / Diagram / Table /
List / Formula / Marking_* / ..., down to Textline > Word. Every stroke is in
exactly one leaf element. It is usable on its own for page-level work.

The shards hold one Textline per entry. Its text is its words' transcriptions
joined by single spaces (the line's own transcription has stray double spaces
and <Symbol/> placeholders), split into char tokens. List bullets (Symbol)
and arrows inside a line are not text, so their strokes are left out. Lines
with an untranscribed word, longer than MAX_TEXT tokens or using a token that
no training line has are skipped.

IAMonDo labels words, not characters, so every chars entry is -1. The
alignment is per element instead; each shard has, besides the usual keys:
  elements        int16   (total points,)  index of each point's element in its line
  element_counts  int32   (lines,)         elements per line; split the next two with it
  element_types   str     (total elements,) "Word" or "Correction" (struck-out
                                           strokes inside a word)
  element_spans   int16   (total elements, 2)  [start, end) token indices of a
                                           Word in the line's text; (-1, -1) for a
                                           Correction, whose text is not in the line
  origins         float32 (lines, 2)       page position (mm) of the line's first point
  units           float32 (lines,)         mm per ink unit for the line
  documents       str     (lines,)         source file, e.g. "114.inkml"
so a point's page position is origins + units * cumsum(offsets * std + mu)[:, :2].

Normalization matches mathwriting_corpus: scaled by the line's median stroke
size, resampled every --spacing units along each stroke, timestamps dropped
(strokes are ordered by start time first).

IAMonDo's documents are split into five folds (0.set .. 4.set). Each fold
becomes one shard, with --held-out last, so Corpus.split (--val-shards 1)
holds it out. Normalization and the charset come from the other folds.
Documents in no fold are skipped.

Usage: python scripts/iamondo_corpus.py [--source data/IAMonDo-db-1.0]
           [--out-dir data/iamondo_corpus] [--held-out 4]
"""
import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np

from ink_corpus import MAX_TEXT
from mathwriting_corpus import resample

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
VALUE = re.compile(r"""([!'"]?)\s*(-?(?:\d+\.?\d*|\.\d+))""")
ANOTO_MM = 0.3  # Anoto dot pitch; a page is 989 x 699 units, i.e. A4


@dataclass
class Element:
    type: str
    transcription: str | None
    traces: list[str] = field(default_factory=list)  # this element's own strokes, not its children's
    children: list["Element"] = field(default_factory=list)

    def all_traces(self):
        return self.traces + [t for c in self.children for t in c.all_traces()]


def decode_trace(text):
    """(n, channels) array from an InkML trace. A value prefixed ' is a first
    difference and " a second difference; an unprefixed value keeps the
    previous one's kind on that channel."""
    points, mode, value, velocity = [], None, None, None
    for chunk in text.strip().split(","):
        tokens = VALUE.findall(chunk)
        if mode is None:
            mode, value, velocity = ["!"] * len(tokens), np.zeros(len(tokens)), np.zeros(len(tokens))
        for c, (prefix, number) in enumerate(tokens):
            mode[c] = prefix or mode[c]
            x = float(number)
            if mode[c] == "!":
                velocity[c], value[c] = x - value[c], x
            elif mode[c] == "'":
                velocity[c] = x
                value[c] += x
            else:
                velocity[c] += x
                value[c] += velocity[c]
        points.append(value.copy())
    return np.array(points)


def read_element(view):
    annotations = {a.get("type"): a.text for a in view.findall("annotation")}
    element = Element(annotations.get("type"), annotations.get("transcription"))
    for child in view.findall("traceView"):
        if child.get("traceDataRef"):
            element.traces.append(child.get("traceDataRef").lstrip("#"))
        else:
            element.children.append(read_element(child))
    return element


def read_document(path):
    """(strokes, root): strokes maps trace id to an (n, 3) array of x, y (mm,
    y down, text reading left to right) and t (s); root is the Document
    element."""
    root = ET.parse(path).getroot()
    # Pens recorded pages in different orientations; the canvasTransform
    # turns each upright.
    matrix = root.find(".//canvasTransform/mapping/matrix")
    turn = np.eye(2)
    if matrix is not None:
        rows = [r.split() for r in matrix.text.strip().strip(",").split(",")]
        turn = np.array([[float(v) for v in r[:2]] for r in rows[:2]])
    strokes = {}
    for trace in root.iter("trace"):
        p = decode_trace(trace.text)
        xy = p[:, :2] @ turn.T * ANOTO_MM
        strokes[trace.get(XML_ID)] = np.stack([xy[:, 0], xy[:, 1], p[:, 2]], axis=1)
    return strokes, read_element(root.find("traceView"))


def textlines(element):
    if element.type == "Textline":
        yield element
    for child in element.children:
        yield from textlines(child)


def line_strokes(line):
    """(text, [(stroke ids, element type, token span)]) for a Textline, or None
    if a word is untranscribed."""
    words = [c for c in line.children if c.type == "Word"]
    if not words or any(not (w.transcription or "").strip() for w in words):
        return None
    parts, groups, start = [], [], 0
    for w in words:
        text = w.transcription.strip()
        groups.append((w.traces, "Word", (start, start + len(text))))
        for c in w.children:
            groups.append((c.all_traces(), c.type, (-1, -1)))
        parts.append(text)
        start += len(text) + 1
    return " ".join(parts), groups


def line_offsets(strokes, groups, spacing):
    """Raw (dx, dy, end_of_stroke) in ink units, each point's element index,
    the first point's page position and mm per unit; None for an unusable line."""
    labelled = sorted(((strokes[t], e) for e, (ids, _, _) in enumerate(groups) for t in ids),
                      key=lambda s: s[0][0, 2])
    sizes = [np.ptp(s[:, :2], axis=0).max() for s, _ in labelled]
    unit = float(np.median(sizes)) if sizes else 0.0
    if unit <= 0:
        return None
    points, eos, labels = [], [], []
    for s, e in labelled:
        r = resample(s[:, :2] / unit, spacing)
        points.append(r)
        eos.append(np.arange(len(r)) == len(r) - 1)
        labels.append(np.full(len(r), e))
    xy = np.concatenate(points)
    origin = xy[0] * unit
    xy -= xy[0]
    d = np.diff(xy, axis=0, prepend=xy[:1])
    offsets = np.concatenate([d, np.concatenate(eos)[:, None]], axis=1).astype(np.float32)
    return offsets, np.concatenate(labels), origin, unit


def convert(args):
    path, spacing, max_points = args
    strokes, root = read_document(path)
    out, skipped = [], 0
    for line in textlines(root):
        parsed = line_strokes(line)
        if parsed is None or len(parsed[0]) > MAX_TEXT:
            skipped += 1
            continue
        text, groups = parsed
        converted = line_offsets(strokes, groups, spacing)
        if converted is None or len(converted[0]) > max_points:
            skipped += 1
            continue
        offsets, labels, origin, unit = converted
        out.append({"text": text, "offsets": offsets, "elements": labels, "origin": origin, "unit": unit,
                    "types": [g[1] for g in groups], "spans": [g[2] for g in groups],
                    "document": os.path.basename(path)})
    return out, skipped


def read_folds(source):
    folds = {}
    for path in sorted(glob.glob(os.path.join(source, "*.set"))):
        fold = int(os.path.basename(path).split(".")[0])
        for name in open(path).read().split():
            folds[name] = fold
    return folds


def write_shard(path, lines, mu, std):
    offsets = [(l["offsets"] - mu) / std for l in lines]
    np.savez(path,
             offsets=np.concatenate(offsets).astype(np.float32),
             chars=np.full(sum(len(o) for o in offsets), -1, dtype=np.int16),
             lengths=np.array([len(o) for o in offsets], dtype=np.int32),
             texts=np.array([l["text"] for l in lines]),
             bias=np.zeros(len(lines), dtype=np.float32),
             rejected=np.int32(0),
             elements=np.concatenate([l["elements"] for l in lines]).astype(np.int16),
             element_counts=np.array([len(l["types"]) for l in lines], dtype=np.int32),
             element_types=np.array([t for l in lines for t in l["types"]]),
             element_spans=np.array([s for l in lines for s in l["spans"]], dtype=np.int16),
             origins=np.array([l["origin"] for l in lines], dtype=np.float32),
             units=np.array([l["unit"] for l in lines], dtype=np.float32),
             documents=np.array([l["document"] for l in lines]))


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to convert again")
    folds = read_folds(args.source)
    if args.held_out not in folds.values():
        raise SystemExit(f"no fold {args.held_out} in {args.source}")
    paths = sorted(p for p in glob.glob(os.path.join(args.source, "*.inkml")) if os.path.basename(p) in folds)
    unassigned = len(glob.glob(os.path.join(args.source, "*.inkml"))) - len(paths)
    with Pool(args.workers) as pool:
        results = pool.map(convert, [(p, args.spacing, args.max_points) for p in paths], chunksize=8)

    by_fold = {f: [] for f in sorted(set(folds.values()))}
    skipped = 0
    for path, (lines, n_skipped) in zip(paths, results):
        by_fold[folds[os.path.basename(path)]].extend(lines)
        skipped += n_skipped
    order = [f for f in by_fold if f != args.held_out] + [args.held_out]
    train = [l for f in order[:-1] for l in by_fold[f]]

    charset = sorted({ch for l in train for ch in l["text"]})
    allowed = set(charset)
    held_out = by_fold[args.held_out]
    by_fold[args.held_out] = [l for l in held_out if set(l["text"]) <= allowed]
    skipped += len(held_out) - len(by_fold[args.held_out])

    xy = np.concatenate([l["offsets"][:, :2] for l in train])
    mu = np.array([*xy.mean(axis=0), 0.0], dtype=np.float32)
    std = np.array([*xy.std(axis=0), 1.0], dtype=np.float32)
    lengths = np.array([len(l["offsets"]) for l in train])
    print(f"points per line p50/p90/p99: {np.percentile(lengths, [50, 90, 99]).round().tolist()}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for k, f in enumerate(order):
        write_shard(os.path.join(args.out_dir, f"shard_{k:05d}.npz"), by_fold[f], mu, std)
        print(f"shard {k}: fold {f}, {len(by_fold[f])} lines", flush=True)
    with open(os.path.join(args.out_dir, "lines.txt"), "w") as f:
        f.write("\n".join(l["text"] for l in train) + "\n")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        # text_height: as in mathwriting_corpus, a rough guess with a symbol about one unit.
        json.dump({"charset": charset, "tokenizer": "char", "mu": mu.tolist(), "std": std.tolist(),
                   "source": "IAMonDo-db-1.0", "folds": order, "held_out": args.held_out,
                   "spacing": args.spacing, "text_height": 1.5, "lines": "lines.txt"}, f, indent=1)
    print(f"{len(charset)} tokens; skipped {skipped} lines and {unassigned} documents in no fold", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="data/IAMonDo-db-1.0")
    ap.add_argument("--out-dir", default="data/iamondo_corpus")
    ap.add_argument("--held-out", type=int, default=4, help="fold written last, as the validation shard")
    ap.add_argument("--spacing", type=float, default=0.1, help="resampling step, in symbol sizes")
    ap.add_argument("--max-points", type=int, default=2000, help="skip longer lines after resampling")
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
