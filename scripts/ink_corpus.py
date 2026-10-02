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


class Corpus:
    def __init__(self, corpus_dir, shards):
        self.meta = json.load(open(os.path.join(corpus_dir, "meta.json")))
        self.charset = self.meta["charset"]
        self.token = char_tokens(self.charset)
        self.tokenizer = self.meta.get("tokenizer", "char")
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

    def batch(self, indices):
        """text (B, T) left-padded tokens, text_len (B,), offsets (B, N, 3)
        and chars (B, N) right-padded with zeros, ink_len (B,)."""
        texts = [self.texts[i] for i in indices]
        n = int(self.lengths[indices].max())
        offsets = torch.zeros(len(indices), n, 3)
        chars = torch.zeros(len(indices), n, dtype=torch.long)
        for b, i in enumerate(indices):
            s, k = self.starts[i], self.lengths[i]
            offsets[b, :k] = torch.from_numpy(self.offsets[s:s + k])
            chars[b, :k] = torch.from_numpy(self.chars[s:s + k].astype(np.int64))
        text, text_len = encode_texts(texts, self.token, self.tokenizer)
        return {"text": text, "text_len": text_len, "offsets": offsets, "chars": chars, "ink_len": torch.from_numpy(self.lengths[indices].astype(np.int64))}

    def batches(self, max_points, rng, pool=64):
        """One epoch of index batches, each holding at most max_points padded
        ink points. Lines are sorted by length within pools of nearby
        batches, so padding stays small but batches still mix lines."""
        order = list(range(len(self)))
        rng.shuffle(order)
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
        rng.shuffle(out)
        return out

