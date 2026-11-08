r"""Convert DeepWriting's extended dataset (the "dataset with timestamps",
data/deepwriting/extended_dataset.zip, read in place) into corpus shards in
the format of scripts/mathwriting_corpus.py, with each point aligned to the
character of the line it draws.

License: CC BY-NC-SA 4.0 with no distribution (one archive copy only). Do not
push this corpus, or a pooled corpus containing it, to W&B or copy it to
other machines; convert it where it is downloaded.

The zip holds one JSON file per IAM-OnDB form and writer (294 writers: the
cleaned IAM-OnDB lines and the authors' own), each with several lines. A line
has its prompt text, timestamped points (an `ev` of 2 ends a stroke) and words
and characters as point-index ranges. Lines are skipped when they have no ink,
are flagged as misspelled (the writer wrote something other than the prompt),
or are longer than MAX_TEXT characters.

The character ranges carry the labels of the authors' recognizer, not the
prompt: punctuation is left out and some characters are misread. They are
aligned to the prompt's characters (case-insensitively) and only matched
characters label their points; every other point gets -1 ("unknown"), as do
all points of lines whose segmentation is flagged invalid.

Every --held-out-every-th writer is the validation shard (the last shard), so
validation is on unseen writers.

Usage: python scripts/deepwriting_corpus.py [--source data/deepwriting/extended_dataset.zip]
           [--out-dir data/deepwriting_corpus]
"""
import argparse
import glob
import json
import os
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from multiprocessing import Pool

import numpy as np

from ink_corpus import MAX_TEXT
from text_corpus import held_out_writers, ink_offsets, split_strokes, write_corpus


def truthy(value):
    return value is True or str(value).lower() == "true"


def char_labels(sample, text, n):
    """Per-point index into text of the character each point draws, -1 where
    it is unknown."""
    labels = np.full(n, -1)
    segments = [c for w in sample.get("wholeword_segments") or [] for c in w.get("chars", [])]
    if not truthy(sample.get("is_char_segmentation_valid")) or any(len(c["char"]) != 1 for c in segments):
        return labels
    positions = [i for i, ch in enumerate(text) if ch != " "]
    written = "".join(c["char"] for c in segments).lower()
    prompt = "".join(text[i] for i in positions).lower()
    for a, b, size in SequenceMatcher(None, written, prompt, autojunk=False).get_matching_blocks():
        for k in range(size):
            for r in segments[a + k]["ranges"]:
                index = np.asarray(r, dtype=int)
                labels[index[(index >= 0) & (index < n)]] = positions[b + k]
    return labels


def read_line(sample, spacing, max_points):
    """A line dict for write_corpus without its writer, or the reason it is skipped."""
    points = sample.get("word_stroke")
    if not isinstance(points, list) or len(points) < 2:
        return "no ink"
    if truthy(sample.get("is_sentence_misspelled")):
        return "misspelled"
    text = " ".join(str(sample.get("word_ascii", "")).split())
    if not text:
        return "no text"
    if len(text) > MAX_TEXT:
        return "too long"
    xy = np.array([[float(p["x"]), float(p["y"])] for p in points])
    ends = np.array([int(float(p["ev"])) == 2 for p in points])
    labels = char_labels(sample, text, len(points))
    strokes, cuts = split_strokes(xy, ends)
    converted = ink_offsets(strokes, np.split(labels, cuts), spacing)
    if converted is None:
        return "no extent"
    offsets, chars = converted[:2]
    if len(offsets) > max_points:
        return "too many points"
    return {"text": text, "offsets": offsets, "chars": chars}


def convert(args):
    zip_path, name, spacing, max_points = args
    writer = name.split("/")[1]
    with zipfile.ZipFile(zip_path) as z:
        samples = json.loads(z.read(name))
    lines, skipped = [], Counter()
    for key in sorted(samples, key=lambda k: int(k.removeprefix("sample")) if k.removeprefix("sample").isdigit() else k):
        line = read_line(samples[key], spacing, max_points)
        if isinstance(line, str):
            skipped[line] += 1
        else:
            lines.append({**line, "writer": writer})
    return lines, skipped


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to convert again")
    with zipfile.ZipFile(args.source) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json") and n.count("/") == 3)
    with Pool(args.workers) as pool:
        results = pool.map(convert, [(args.source, n, args.spacing, args.max_points) for n in names], chunksize=8)
    lines = [l for ls, _ in results for l in ls]
    skipped = sum((s for _, s in results), Counter())
    aligned = sum((l["chars"] >= 0).mean() for l in lines) / len(lines)
    print(f"{len(lines)} lines from {len(names)} files; skipped {dict(skipped)}; "
          f"{aligned:.0%} of points aligned to a character on average", flush=True)

    valid_writers = held_out_writers({l["writer"] for l in lines}, args.held_out_every)
    train = [l for l in lines if l["writer"] not in valid_writers]
    valid = [l for l in lines if l["writer"] in valid_writers]
    dropped = write_corpus(args.out_dir, train, valid, "deepwriting-extended", args.spacing, args.lines_per_shard)
    print(f"dropped {dropped} validation lines with characters no training line has", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="data/deepwriting/extended_dataset.zip")
    ap.add_argument("--out-dir", default="data/deepwriting_corpus")
    ap.add_argument("--held-out-every", type=int, default=20, help="every n-th writer is held out")
    ap.add_argument("--spacing", type=float, default=0.1, help="resampling step, in symbol sizes")
    ap.add_argument("--max-points", type=int, default=2000, help="skip longer lines after resampling")
    ap.add_argument("--lines-per-shard", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
