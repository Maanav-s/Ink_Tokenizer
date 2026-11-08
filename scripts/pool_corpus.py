r"""Pool the MathWriting corpus (scripts/mathwriting_corpus.py) with corpora
of human handwritten text (scripts/iamondo_corpus.py, deepwriting_corpus.py,
brush_corpus.py) into one corpus, so one recognizer reads formulas and text.

A text line `some words` becomes `\text{some words}`, as mixed_corpus.py does
for teacher lines; lines too long to fit MAX_TEXT once wrapped, or with
characters the latex tokenizer would not keep as one token each (a
backslash), are dropped. Per-point character labels are shifted past `\text{`.

All these corpora are normalized the same way (scale by median stroke size,
resample by arc length), so text inks only need MathWriting's mean and std;
the corpora must agree on spacing and text_height. MathWriting's shards are
used as they are (symlinked), and its vocabulary comes first with the new
tokens appended, as in mixed_corpus.py.

The last shard is the validation set: every corpus's last shard. Shards
written here have a `sources` entry per line (the corpus's directory name
without "_corpus"), so train_recognizer.py scores each source on its own, and
a `writers` entry, "source/id" for corpora that know who wrote a line
(DeepWriting, BRUSH) and "" otherwise, for the style encoder
(scripts/style_flow_model.py).

--fonts adds up to --font-lines of a font corpus's rendered lines
(scripts/font_corpus.py, built on the same MathWriting corpus, so in the same
normalization) as training shards, each font a writer. Its font_valid.npz is
left out, so validation stays human ink.

DeepWriting's license forbids distribution: a pool containing it must not be
pushed to W&B or copied elsewhere.

Usage: python scripts/pool_corpus.py [--mathwriting data/mathwriting_corpus]
           [--text data/iamondo_corpus data/deepwriting_corpus data/brush_corpus]
           [--out-dir data/pooled_corpus] [--fonts data/font_corpus [--font-lines 20000]]
"""
import argparse
import glob
import json
import os

import numpy as np

from ink_corpus import MAX_TEXT, split_text

SHARD_KEYS = ("offsets", "chars", "lengths", "texts", "bias")
WRAPPER_TOKENS = 3  # \text, { and }
LABEL_SHIFT = 2     # tokens before the line's first character


def wrap_shard(path, src_meta, meta, source):
    """A text corpus shard in the pool's normalization, its lines wrapped in
    \text{}, plus how many lines were dropped."""
    # Indexing an NpzFile decompresses the whole array again, so load each one once.
    d = dict(np.load(path))
    src_mu, src_std = np.array(src_meta["mu"]), np.array(src_meta["std"])
    mu, std = np.array(meta["mu"]), np.array(meta["std"])
    out = {k: [] for k in SHARD_KEYS}
    start, dropped, keep = 0, 0, []
    for i, (text, n) in enumerate(zip(d["texts"], d["lengths"])):
        offsets, chars = d["offsets"][start:start + n], d["chars"][start:start + n]
        start += n
        wrapped = "\\text{" + str(text) + "}"
        if len(split_text(wrapped, "latex")) != len(text) + WRAPPER_TOKENS or len(text) + WRAPPER_TOKENS > MAX_TEXT:
            dropped += 1
            continue
        keep.append(i)
        out["offsets"].append(((offsets * src_std + src_mu) - mu) / std)
        out["chars"].append(np.where(chars >= 0, chars + LABEL_SHIFT, -1))
        out["lengths"].append(n)
        out["texts"].append(wrapped)
    kept = len(out["texts"])
    return {"offsets": np.concatenate(out["offsets"]).astype(np.float32) if kept else np.zeros((0, 3), np.float32),
            "chars": np.concatenate(out["chars"]).astype(np.int16) if kept else np.zeros(0, np.int16),
            "lengths": np.array(out["lengths"], dtype=np.int32),
            "texts": np.array(out["texts"], dtype=str),
            "bias": np.zeros(kept, dtype=np.float32),
            "sources": np.array([source] * kept, dtype=str),
            "writers": np.array([f"{source}/{w}" for w in d["writers"][keep]] if "writers" in d else [""] * kept,
                                dtype=str)}, dropped


def main(args):
    if glob.glob(os.path.join(args.out_dir, "shard_*.npz")):
        raise SystemExit(f"{args.out_dir} already has shards; remove them to pool again")
    meta = json.load(open(os.path.join(args.mathwriting, "meta.json")))
    math_shards = sorted(glob.glob(os.path.join(args.mathwriting, "shard_*.npz")))
    if len(math_shards) < 2:
        raise SystemExit(f"need at least 2 shards in {args.mathwriting}")
    texts = []
    for corpus in args.text:
        src_meta = json.load(open(os.path.join(corpus, "meta.json")))
        for key in ("spacing", "text_height"):
            if src_meta[key] != meta[key]:
                raise SystemExit(f"{corpus} has {key} {src_meta[key]}, {args.mathwriting} has {meta[key]}")
        shards = sorted(glob.glob(os.path.join(corpus, "shard_*.npz")))
        if len(shards) < 2:
            raise SystemExit(f"need at least 2 shards in {corpus}")
        texts.append((os.path.basename(os.path.normpath(corpus)).removesuffix("_corpus"), src_meta, shards))
    os.makedirs(args.out_dir, exist_ok=True)

    def shard_path(index):
        return os.path.join(args.out_dir, f"shard_{index:05d}.npz")

    count = 0
    for path in math_shards[:-1]:
        os.symlink(os.path.relpath(path, args.out_dir), shard_path(count))
        count += 1
    lines, charset = [], set()
    held_out = [{k: np.load(math_shards[-1])[k] for k in SHARD_KEYS}]
    held_out[0]["sources"] = np.array(["mathwriting"] * len(held_out[0]["texts"]), dtype=str)
    held_out[0]["writers"] = np.array([""] * len(held_out[0]["texts"]), dtype=str)
    for source, src_meta, shards in texts:
        kept, dropped = 0, 0
        for path in shards[:-1]:
            shard, n = wrap_shard(path, src_meta, meta, source)
            np.savez(shard_path(count), **shard, rejected=np.int32(0))
            count += 1
            lines.extend(shard["texts"])
            kept, dropped = kept + len(shard["texts"]), dropped + n
        valid, n = wrap_shard(shards[-1], src_meta, meta, source)
        held_out.append(valid)
        charset |= set(src_meta["charset"])
        print(f"{source}: {kept} training and {len(valid['texts'])} validation lines; "
              f"dropped {dropped + n} (too long once wrapped, or a backslash)", flush=True)
    font_meta, font_charset = None, set()
    if args.fonts:
        font_meta = json.load(open(os.path.join(args.fonts, "meta.json")))
        for key in ("mu", "std", "spacing"):
            if font_meta[key] != meta[key]:
                raise SystemExit(f"{args.fonts} has {key} {font_meta[key]}, {args.mathwriting} has {meta[key]}")
        # The font corpus symlinks its base's shards; its own are the rendered ones.
        rendered = [p for p in sorted(glob.glob(os.path.join(args.fonts, "shard_*.npz"))) if not os.path.islink(p)]
        added = 0
        for path in rendered:
            if added >= args.font_lines:
                break
            d = dict(np.load(path))
            if "writers" not in d:
                raise SystemExit(f"{path} has no writers; rebuild {args.fonts} with the current font_corpus.py")
            k = min(len(d["texts"]), args.font_lines - added)
            n_points = int(d["lengths"][:k].sum())
            np.savez(shard_path(count), rejected=np.int32(0), offsets=d["offsets"][:n_points],
                     chars=d["chars"][:n_points], lengths=d["lengths"][:k], texts=d["texts"][:k],
                     bias=d["bias"][:k], writers=d["writers"][:k], sources=np.array(["font"] * k, dtype=str))
            count += 1
            lines.extend(d["texts"][:k])
            added += k
        font_charset = set(font_meta["charset"])
        print(f"font: {added} training lines from {args.fonts}", flush=True)
    np.savez(shard_path(count), rejected=np.int32(0),
             **{k: np.concatenate([h[k] for h in held_out]) for k in (*SHARD_KEYS, "sources", "writers")})
    counts = [f"{h['sources'][0]} {len(h['texts'])}" for h in held_out if len(h["texts"])]
    print(f"validation: {', '.join(counts)}", flush=True)

    math_lines = open(os.path.join(args.mathwriting, meta["lines"])).read()
    with open(os.path.join(args.out_dir, "lines.txt"), "w") as f:
        f.write(math_lines + "\n".join(lines) + "\n")
    # Font tokens last, so a pool with fonts extends the same pool's vocabulary without them.
    new_tokens = [t for t in dict.fromkeys(["\\text", "{", "}", *sorted(charset), *sorted(font_charset)])
                  if t not in meta["charset"]]
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({**meta, "charset": meta["charset"] + new_tokens,
                   "source": [meta["source"], *(m["source"] for _, m, _ in texts),
                              *([font_meta["source"]] if font_meta else [])], "lines": "lines.txt"},
                  f, indent=1)
    print(f"{len(meta['charset'])} MathWriting tokens + {new_tokens}", flush=True)
    print(f"wrote {count} train shards and 1 validation shard to {args.out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mathwriting", default="data/mathwriting_corpus")
    ap.add_argument("--text", nargs="+",
                    default=["data/iamondo_corpus", "data/deepwriting_corpus", "data/brush_corpus"])
    ap.add_argument("--out-dir", default="data/pooled_corpus")
    ap.add_argument("--fonts", help="a font corpus (scripts/font_corpus.py) whose rendered lines to add")
    ap.add_argument("--font-lines", type=int, default=20000,
                    help="at most this many font lines; a font corpus renders every MathWriting label per font")
    main(ap.parse_args())
