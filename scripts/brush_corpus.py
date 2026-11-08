r"""Convert BRUSH (data/brush/refined_BRUSH.zip, read in place) into corpus
shards in the format of scripts/mathwriting_corpus.py, with each point aligned
to the character it draws.

License: non-commercial research only; cite Kotani et al., ECCV 2020.

The zip holds BRUSH/<writer>/<drawing> for 170 writers: single lines written
on tablets, of random words and symbols rather than prose. Only the original
drawings are read (not the _resample20/_resample25 copies, nor the
undocumented *_inter files). Each is a pickle of [sentence, drawing (n, 3) of
x, y, end_of_stroke, label (n, len(sentence)) one-hot], so every point is
labelled with its character's index. Pickles can run code when loaded, so
they are read with an unpickler that only rebuilds numpy arrays.

Every --held-out-every-th writer is the validation shard (the last shard), so
validation is on unseen writers.

Usage: python scripts/brush_corpus.py [--source data/brush/refined_BRUSH.zip]
           [--out-dir data/brush_corpus]
"""
import argparse
import glob
import io
import os
import pickle
import re
import zipfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
from numpy._core.multiarray import _reconstruct

from ink_corpus import MAX_TEXT
from text_corpus import held_out_writers, ink_offsets, split_strokes, write_corpus

DRAWING = re.compile(r"BRUSH/(\d+)/(\d+)$")
ARRAY_GLOBALS = {("numpy.core.multiarray", "_reconstruct"): _reconstruct,
                 ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
                 ("numpy", "ndarray"): np.ndarray,
                 ("numpy", "dtype"): np.dtype}


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ARRAY_GLOBALS:
            return ARRAY_GLOBALS[module, name]
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def read_line(data, spacing, max_points):
    """A line dict for write_corpus without its writer, or the reason it is skipped."""
    sentence, drawing, label = ArrayUnpickler(io.BytesIO(data)).load()
    sentence = str(sentence)
    text = sentence.strip()
    if not text:
        return "no text"
    if len(text) > MAX_TEXT:
        return "too long"
    lead = len(sentence) - len(sentence.lstrip())
    labels = np.where(label.max(axis=1) > 0, label.argmax(axis=1) - lead, -1)
    labels[(labels < 0) | (labels >= len(text))] = -1
    strokes, cuts = split_strokes(np.asarray(drawing[:, :2], dtype=float), drawing[:, 2] > 0.5)
    converted = ink_offsets(strokes, np.split(labels, cuts), spacing)
    if converted is None:
        return "no extent"
    offsets, chars = converted[:2]
    if len(offsets) > max_points:
        return "too many points"
    return {"text": text, "offsets": offsets, "chars": chars}


def convert(args):
    zip_path, names, spacing, max_points = args
    lines, skipped = [], Counter()
    with zipfile.ZipFile(zip_path) as z:
        for name in names:
            line = read_line(z.read(name), spacing, max_points)
            if isinstance(line, str):
                skipped[line] += 1
            else:
                lines.append({**line, "writer": DRAWING.match(name).group(1)})
    return lines, skipped


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to convert again")
    with zipfile.ZipFile(args.source) as z:
        names = sorted(n for n in z.namelist() if DRAWING.match(n))
    chunks = [names[i:i + 256] for i in range(0, len(names), 256)]
    with Pool(args.workers) as pool:
        results = pool.map(convert, [(args.source, c, args.spacing, args.max_points) for c in chunks])
    lines = [l for ls, _ in results for l in ls]
    skipped = sum((s for _, s in results), Counter())
    print(f"{len(lines)} lines from {len(names)} drawings; skipped {dict(skipped)}", flush=True)

    valid_writers = held_out_writers({l["writer"] for l in lines}, args.held_out_every)
    train = [l for l in lines if l["writer"] not in valid_writers]
    valid = [l for l in lines if l["writer"] in valid_writers]
    dropped = write_corpus(args.out_dir, train, valid, "brush-refined", args.spacing, args.lines_per_shard)
    print(f"dropped {dropped} validation lines with characters no training line has", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="data/brush/refined_BRUSH.zip")
    ap.add_argument("--out-dir", default="data/brush_corpus")
    ap.add_argument("--held-out-every", type=int, default=20, help="every n-th writer is held out")
    ap.add_argument("--spacing", type=float, default=0.1, help="resampling step, in symbol sizes")
    ap.add_argument("--max-points", type=int, default=2000, help="skip longer lines after resampling")
    ap.add_argument("--lines-per-shard", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
