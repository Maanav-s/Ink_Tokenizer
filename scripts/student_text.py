"""Text items written by the text-to-ink student, a few characters at a time, for render_inkml --student.

render_inkml lays a text item out in its font first; that gives each character's place on the page,
which the layout's collision checks were run against. Each word is cut into chunks of at most
CHUNK_MAX characters, the student writes every chunk on the page in one batch, and a chunk's ink
replaces its glyphs, scaled into their bounding box. The student labels each point with the token it
draws, so every stroke goes to one character's unit. Characters the student has no token for, and
crossed-out attempts, keep their font strokes.
"""
import unicodedata

import numpy as np
import torch

from ink_corpus import PAD, char_tokens
from render_inkml import Unit
from sample_student import load_model

CHUNK_MAX = 4
BATCH = 64

SYMBOLS = {"→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow", "⇒": r"\Rightarrow",
           "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx", "±": r"\pm", "·": r"\cdot",
           "×": r"\times", "∞": r"\infty", "∫": r"\int", "∑": r"\sum", "∂": r"\partial", "∆": r"\Delta",
           "°": r"\circ", "∈": r"\in", "⊕": r"\oplus", "'": r"\prime", "#": r"\#", "%": r"\%", "&": r"\&",
           "_": r"\_", "{": r"\{", "}": r"\}", "\\": r"\backslash"}


def chunks(word):
    """A word cut into ceil(n / CHUNK_MAX) nearly equal pieces: 5 -> 3 + 2, 7 -> 4 + 3."""
    k = -(-len(word) // CHUNK_MAX)
    size, extra = divmod(len(word), k)
    out, i = [], 0
    for j in range(k):
        n = size + (j < extra)
        out.append(word[i:i + n])
        i += n
    return out


class StudentText:
    def __init__(self, checkpoint, device="cpu", seed=0):
        self.model, ckpt = load_model(checkpoint, device)
        self.token = char_tokens(ckpt["charset"])
        self.mu, self.std = np.array(ckpt["meta"]["mu"]), np.array(ckpt["meta"]["std"])
        self.device, self.seed = device, seed
        self.ink = {}  # (item id, first char index) -> (strokes, per-stroke char offset in the chunk)

    def tokens(self, ch):
        if ch in self.token:
            return ch
        name = unicodedata.name(ch, "")
        if name.startswith("GREEK"):
            ch = "\\" + name.split()[-1].lower() if "SMALL" in name else "\\" + name.split()[-1].capitalize()
        else:
            ch = SYMBOLS.get(ch, ch)
        return ch if ch in self.token else None

    def runs(self, text):
        """(first char index, chunk) for every chunk of text the student can write."""
        out, i = [], 0
        while i < len(text):
            if self.tokens(text[i]) is None or text[i] == " ":
                i += 1
                continue
            j = i
            while j < len(text) and text[j] != " " and self.tokens(text[j]) is not None:
                j += 1
            for piece in chunks(text[i:j]):
                out.append((i, piece))
                i += len(piece)
        return out

    @torch.no_grad()
    def prepare(self, layout):
        """Write every chunk of the layout's text items."""
        jobs = [((eid, i), piece) for eid, item in layout["items"].items() if item["kind"] == "text"
                for i, piece in self.runs(item["text"])]
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        for b in range(0, len(jobs), BATCH):
            batch = jobs[b:b + BATCH]
            toks = [[self.token[self.tokens(ch)] for ch in piece] for _, piece in batch]
            width = max(map(len, toks))
            text = torch.tensor([[PAD] * (width - len(t)) + t for t in toks], device=self.device)  # left-padded
            text_len = torch.tensor([len(t) for t in toks], device=self.device)
            for (key, _), (offsets, labels, _) in zip(batch, self.model.sample(text, text_len, 0.0,
                                                                              generator=gen)):
                offsets, labels = offsets.float().numpy(), labels.numpy()
                xy = np.cumsum(offsets[:, :2] * self.std[:2] + self.mu[:2], axis=0)
                ends = np.flatnonzero(offsets[:, 2] > 0.5) + 1
                strokes, owners = [], []
                for pts, lab in zip(np.split(xy, ends), np.split(labels, ends)):
                    if len(pts):
                        strokes.append(pts)
                        owners.append(int(np.bincount(lab[lab >= 0]).argmax()) if (lab >= 0).any() else 0)
                self.ink[key] = (strokes, owners)
        print(f"student wrote {len(jobs)} chunks")

    def replace(self, units, item, eid):
        by_char = {u.label["char_index"]: u for u in units
                   if "char_index" in u.label and u.label.get("status") is None}
        replaced, chunk_runs = {}, {}  # char index -> its chunk's start; chunk start -> stroke runs
        for start, piece in self.runs(item["text"]):
            glyphs = [by_char[i] for i in range(start, start + len(piece)) if i in by_char]
            pts = np.array([p for u in glyphs for st in u.strokes for p in st])
            if (eid, start) not in self.ink or not len(pts):
                continue
            strokes, owners = self.ink[(eid, start)]
            ink = np.concatenate(strokes)
            h = item["text_height"]
            # Single characters like "-" or "." have nearly flat boxes, so the fit has floors.
            (fx0, fy0), (fx1, fy1) = pts.min(0), pts.max(0)
            (ix0, iy0), (ix1, iy1) = ink.min(0), ink.max(0)
            fw, fh = max(fx1 - fx0, 0.15 * h), max(fy1 - fy0, 0.15 * h)
            s = min(fw / max(ix1 - ix0, 1e-6), fh / max(iy1 - iy0, 1e-6))
            centre_f, centre_i = np.array([fx0 + fx1, fy0 + fy1]) / 2, np.array([ix0 + ix1, iy0 + iy1]) / 2
            # Runs of strokes in the order the student wrote them, so a t crossed or an i dotted after
            # the rest of the chunk stays late; such a character then has two units with one label.
            runs = []
            for st, k in zip(strokes, owners):
                i = start + min(k, len(piece) - 1)
                pts_page = [tuple(p) for p in (st - centre_i) * s + centre_f]
                if runs and runs[-1][0] == i:
                    runs[-1][1].append(pts_page)
                else:
                    runs.append((i, [pts_page]))
            chunk_runs[start] = runs
            for k in range(len(piece)):
                replaced[start + k] = start
        out, done = [], set()
        for u in units:
            i = u.label.get("char_index")
            if i in replaced and u.label.get("status") is None:
                start = replaced[i]
                if start not in done:
                    done.add(start)
                    out += [Unit(by_char[c].label, run) for c, run in chunk_runs[start] if c in by_char]
            else:
                out.append(u)
        return out
