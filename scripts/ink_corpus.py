r"""Load corpus shards (scripts/generate_teacher_corpus.py,
scripts/mathwriting_corpus.py) for training.

Model-agnostic: it yields padded batches of text and normalized offsets with
per-point character alignment (-1 where it is unknown), and leaves sequence
layout to the model.

Text is split into tokens by the corpus's tokenizer (meta.json "tokenizer"):
"char" (the default) makes every character a token; "latex" also keeps each
LaTeX command (\frac, \alpha), control symbol (\\, \{) and environment
marker (\begin{matrix}) whole. "charset" in meta.json is the token list.
"""
import glob
import json
import os
import re

import numpy as np
import torch

PAD, SEP = 0, 1  # text token ids; token i of the charset is i + 2
MAX_TEXT = 50    # tokens; matches make_corpus_lines.MAX_CHARS
LATEX_TOKEN = re.compile(r"\\(?:begin|end)\{[A-Za-z*]+\}|\\[A-Za-z]+|\\.|.", re.S)


def split_text(text, tokenizer="char"):
    return LATEX_TOKEN.findall(text) if tokenizer == "latex" else list(text)


def char_tokens(charset):
    return {ch: i + 2 for i, ch in enumerate(charset)}


def encode_texts(texts, token, tokenizer="char"):
    """Left-padded token ids (B, T), so every row's last id is its last token,
    and each row's token count (B,)."""
    pieces = [split_text(t, tokenizer) for t in texts]
    width = max(len(p) for p in pieces)
    out = torch.full((len(texts), width), PAD, dtype=torch.long)
    for b, p in enumerate(pieces):
        if p:
            out[b, width - len(p):] = torch.tensor([token[tok] for tok in p])
    return out, torch.tensor([len(p) for p in pieces])


def resample_ink(offsets, chars, n, mu, std):
    """One ink's normalized offsets (k, 3) and token labels (k,) -> the same
    ink drawn with n points. Each stroke keeps a share of n in proportion to
    its points, and at least one, so no stroke (a dot, say) disappears; its
    points are spread evenly along it by linear interpolation, which keeps
    arc-length spacing for inks already resampled by arc length. Each new
    point takes the label of the nearest original one."""
    xy = np.cumsum(offsets[:, :2] * std + mu, axis=0)
    ends = np.flatnonzero(offsets[:, 2] > 0.5)
    if not len(ends) or ends[-1] != len(offsets) - 1:
        ends = np.append(ends, len(offsets) - 1)
    starts = np.concatenate([[0], ends[:-1] + 1])
    sizes = ends - starts + 1
    n = max(n, len(sizes))
    counts = np.maximum(1, np.round(sizes * n / len(offsets))).astype(int)
    # Rounding and the one-point minimum leave the total a few points off n.
    while counts.sum() != n:
        counts[np.argmax(counts)] += 1 if counts.sum() < n else -1
    new_xy, new_chars, pen = [], [], []
    for s, k, m in zip(starts, sizes, counts):
        t = np.linspace(0, k - 1, m)
        index = np.arange(k)
        new_xy.append(np.stack([np.interp(t, index, xy[s:s + k, 0]), np.interp(t, index, xy[s:s + k, 1])], axis=1))
        new_chars.append(chars[s:s + k][np.rint(t).astype(int)])
        pen.append(np.arange(m) == m - 1)
    new_xy = np.concatenate(new_xy)
    raw = np.diff(new_xy, axis=0, prepend=np.zeros((1, 2)))
    new_offsets = np.concatenate([(raw - mu) / std, np.concatenate(pen)[:, None]], axis=1)
    return new_offsets.astype(offsets.dtype), np.concatenate(new_chars)


class Corpus:
    def __init__(self, corpus_dir, shards):
        self.meta = json.load(open(os.path.join(corpus_dir, "meta.json")))
        self.charset = self.meta["charset"]
        self.token = char_tokens(self.charset)
        self.tokenizer = self.meta.get("tokenizer", "char")
        self.shards = list(shards)
        offsets, chars, lengths, bias, self.texts = [], [], [], [], []
        for path in shards:
            d = np.load(path)
            offsets.append(d["offsets"])
            chars.append(d["chars"])
            lengths.append(d["lengths"])
            bias.append(d["bias"])
            self.texts.extend(str(t) for t in d["texts"])
        self.offsets = np.concatenate(offsets)
        self.chars = np.concatenate(chars)
        self.lengths = np.concatenate(lengths)
        self.bias = np.concatenate(bias)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.text_lengths = [len(split_text(text, self.tokenizer)) for text in self.texts]

    @classmethod
    def split(cls, corpus_dir, val_shards=1, names=None):
        """The last val_shards shards (by name) are the validation set. names
        limits the corpus to those shard files, e.g. one artifact version's."""
        if names is None:
            shards = sorted(glob.glob(os.path.join(corpus_dir, "shard_*.npz")))
        else:
            shards = [os.path.join(corpus_dir, n) for n in sorted(names)]
        if len(shards) <= val_shards:
            raise ValueError(f"need more than {val_shards} shards in {corpus_dir}, found {len(shards)}")
        return cls(corpus_dir, shards[:-val_shards]), cls(corpus_dir, shards[-val_shards:])

    def __len__(self):
        return len(self.texts)

    def batch(self, indices, stretch=None):
        """text (B, T) left-padded tokens, text_len (B,), offsets (B, N, 3)
        and chars (B, N) right-padded with zeros, ink_len (B,).

        stretch, one factor per line, resamples each ink to that many times
        its points (resample_ink); the batch then also has true_len, the
        lengths before resampling."""
        texts = [self.texts[i] for i in indices]
        inks = [(self.offsets[s:s + k], self.chars[s:s + k])
                for s, k in zip(self.starts[indices], self.lengths[indices])]
        true_len = torch.from_numpy(self.lengths[indices].astype(np.int64))
        if stretch is not None:
            mu, std = np.array(self.meta["mu"][:2]), np.array(self.meta["std"][:2])
            inks = [resample_ink(off, ch, max(1, round(len(off) * f)), mu, std)
                    for (off, ch), f in zip(inks, stretch)]
        lengths = [len(off) for off, _ in inks]
        offsets = torch.zeros(len(indices), max(lengths), 3)
        chars = torch.zeros(len(indices), max(lengths), dtype=torch.long)
        for b, (off, ch) in enumerate(inks):
            offsets[b, :len(off)] = torch.from_numpy(off)
            chars[b, :len(off)] = torch.from_numpy(ch.astype(np.int64))
        text, text_len = encode_texts(texts, self.token, self.tokenizer)
        out = {"text": text, "text_len": text_len, "offsets": offsets, "chars": chars,
               "ink_len": torch.tensor(lengths)}
        if stretch is not None:
            out["true_len"] = true_len
        return out

    def batches(self, max_points, rng, pool=64, equal_text_lengths=False):
        """One epoch of index batches, each holding at most max_points padded
        ink points. Lines are sorted by length within pools of nearby
        batches. Optionally group by text token count before ink bucketing."""
        if equal_text_lengths:
            by_length = {}
            for index, length in enumerate(self.text_lengths):
                by_length.setdefault(length, []).append(index)
            out = []
            for length in sorted(by_length):
                order = by_length[length]
                rng.shuffle(order)
                out.extend(self.pack_batches(order, max_points, pool))
            rng.shuffle(out)
            return out
        order = list(range(len(self)))
        rng.shuffle(order)
        out = self.pack_batches(order, max_points, pool)
        rng.shuffle(out)
        return out

    def pack_batches(self, order, max_points, pool):
        out = []
        avg = int(self.lengths.mean())
        chunk = pool * max(1, max_points // avg)
        for c in range(0, len(order), chunk):
            group = sorted(order[c:c + chunk], key=lambda i: self.lengths[i])
            batch, longest = [], 0
            for i in group:
                longest = max(longest, int(self.lengths[i]))
                if batch and longest * (len(batch) + 1) > max_points:
                    out.append(batch)
                    batch, longest = [], int(self.lengths[i])
                batch.append(i)
            if batch:
                out.append(batch)
        return out
