"""Text to ink with the pretrained teacher network (scripts/teacher_model.py).

This is the narrow interface from docs/synthetic_data.md: text plus a style
in, strokes out, in writing order, with every point labelled with the
character it draws. The student models implement the same interface on top
of clean_sample() and to_line().

A style is one writer: a seed, a bias and a priming line. The writer first
writes the priming line unprimed; every later line is primed with that ink,
so all of a writer's lines share one hand. The priming ink also sets the
writer's scale: the teacher writes in IAM-OnDB whiteboard units, and the
middle 96% of the priming line's height is mapped to `text_height` mm.
PRIME_TEXT has ascenders and descenders, so that height spans roughly the
descender line to the cap line, the same text height that scripts/fonts.py
uses.

Output coordinates are in mm, x right and y up, with the line starting at x=0
and the bottom of its middle 96% at y=0. There are no timestamps; placing and
timing lines on a page is left to the renderer.

Usage: python scripts/text_to_ink.py "Some text" ["More lines" ...]
           [--seed N] [--bias B] [--out preview.png]
The preview colours each point by the character it is aligned to.
"""
import argparse
import colorsys

import numpy as np
import torch

from teacher_model import Teacher, default_checkpoint_dir

PRIME_TEXT = "Quick brown fox jumps high"
TEXT_HEIGHT = 40.0  # mm, a typical whiteboard text height


def to_points(ink):
    """Denormalized (dx, dy, eos) -> absolute (x, y up) and end-of-stroke flags."""
    xy = np.cumsum(ink[:, :2], axis=0)
    xy[:, 1] *= -1  # IAM-OnDB y points down
    return xy, ink[:, 2] > 0.5


def plausible(text, ink):
    """Reject the teacher's failure modes that its own end test misses. ink is
    denormalized, in IAM-OnDB units. The bounds come from eyeballing samples:
    good lines are 300-1000 units tall and 100-250 units per character;
    failures scribble vertically or stall in one spot.
    """
    if len(ink) < 2:
        return False
    xy, eos = to_points(ink)
    lo, hi = np.percentile(xy[:, 1], [2, 98])
    per_char = np.ptp(xy[:, 0]) / max(len(text), 1)
    # Even joined-up cursive lifts the pen between words.
    return hi - lo < 1200 and 60 < per_char < 400 and eos.sum() >= len(text.split())


def clean_sample(text, offsets, char, finished, mu, std):
    """Tidy one sampled line in normalized offset space.

    Trailing strokes drawn after the window passed the end of the text
    (usually a stray dot) are cut, and the alignment is clipped to the text.
    Returns (offsets, char, ok); ok is False when the line should be rejected.
    """
    offsets, char = np.asarray(offsets, dtype=np.float32), np.asarray(char)
    ends = np.flatnonzero(offsets[:, 2] > 0.5)
    starts = np.concatenate([[0], ends + 1])
    keep = len(offsets)
    for s in starts[::-1]:
        if s < keep and char[s] >= len(text):
            keep = s
        elif s < keep:
            break
    offsets, char = offsets[:keep], np.clip(char[:keep], 0, len(text) - 1)
    if len(offsets):
        offsets[-1, 2] = 1.0
    ok = finished and plausible(text, offsets * std + mu)
    return offsets, char.astype(np.int16), ok


def to_line(text, offsets, char, mu, std, scale, ok=True):
    """Normalized offsets -> {"text", "strokes": [[(x, y)]] in mm,
    "chars": [[index into text]] parallel to strokes, "ok"}."""
    xy, eos = to_points(np.asarray(offsets) * std + mu)
    xy *= scale
    if len(xy):
        xy -= [xy[:, 0].min(), np.percentile(xy[:, 1], 2)]
    strokes, chars, start = [], [], 0
    for i in range(len(xy)):
        if eos[i] or i == len(xy) - 1:
            strokes.append([(float(x), float(y)) for x, y in xy[start:i + 1]])
            chars.append([int(c) for c in char[start:i + 1]])
            start = i + 1
    return {"text": text, "strokes": strokes, "chars": chars, "ok": ok}


def line_scale(ink, text_height=TEXT_HEIGHT):
    """mm per ink unit, from a line with ascenders and descenders."""
    xy, _ = to_points(ink)
    lo, hi = np.percentile(xy[:, 1], [2, 98])
    return text_height / (hi - lo)


class Writer:
    def __init__(self, teacher, seed=0, bias=0.75, text_height=TEXT_HEIGHT, prime_text=PRIME_TEXT, tries=20):
        """A bad priming line ruins every line primed with it, so priming is
        retried with derived seeds until one is usable."""
        self.teacher, self.seed, self.bias = teacher, seed, bias
        mu, std = teacher.mu.numpy(), teacher.std.numpy()
        for k in range(tries):
            s = teacher.sample([prime_text], bias=bias, seed=seed * tries + k)[0]
            offsets, _, ok = clean_sample(prime_text, s["offsets"], s["char"], s["finished"], mu, std)
            if ok:
                break
        else:
            raise RuntimeError(f"no usable priming line for seed={seed} bias={bias}")
        self.prime = (prime_text, torch.from_numpy(offsets))
        self.scale = line_scale(offsets * std + mu, text_height)

    def write(self, lines, seed=0):
        """One to_line() dict per line. ok=False means the teacher lost its
        place or scribbled and the line should be rejected. The teacher can
        still misspell a word with ok=True."""
        mu, std = self.teacher.mu.numpy(), self.teacher.std.numpy()
        out = []
        for line, s in zip(lines, self.teacher.sample(lines, self.bias, seed, self.prime)):
            offsets, char, ok = clean_sample(line, s["offsets"], s["char"], s["finished"], mu, std)
            out.append(to_line(line, offsets, char, mu, std, self.scale, ok))
        return out


def preview(lines, path, px_per_mm=2.0, line_gap=1.6):
    """Draw lines one under another, each point coloured by its character.
    Rejected lines are drawn dimmed."""
    from PIL import Image, ImageDraw

    pad = 10
    pts = [p for ln in lines for st in ln["strokes"] for p in st] or [(1.0, 1.0)]
    height = max(p[1] for p in pts)
    width = max(p[0] for p in pts)
    row = height * line_gap
    img = Image.new("RGB", (int(width * px_per_mm) + 2 * pad, int(row * len(lines) * px_per_mm) + 2 * pad), "white")
    draw = ImageDraw.Draw(img)
    for r, ln in enumerate(lines):
        y0 = pad + (r * row + height) * px_per_mm
        for stroke, chars in zip(ln["strokes"], ln["chars"]):
            xy = [(pad + x * px_per_mm, y0 - y * px_per_mm) for x, y in stroke]
            for (a, b), c in zip(zip(xy, xy[1:]), chars[1:]):
                rgb = colorsys.hsv_to_rgb((c * 0.61) % 1.0, 0.8, 0.8 if ln["ok"] else 0.4)
                draw.line([a, b], fill=tuple(int(v * 255) for v in rgb), width=2)
            if len(xy) == 1:
                draw.point(xy[0], fill="black")
    img.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lines", nargs="+")
    ap.add_argument("--seed", type=int, default=0, help="writer style seed")
    ap.add_argument("--bias", type=float, default=0.75, help="0 = most varied, higher = neater")
    ap.add_argument("--out", default="text_to_ink_preview.png")
    ap.add_argument("--checkpoint", default=default_checkpoint_dir())
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(4)
    writer = Writer(Teacher(args.checkpoint, args.device), seed=args.seed, bias=args.bias)
    result = writer.write(args.lines, seed=args.seed)
    for ln in result:
        n = sum(len(s) for s in ln["strokes"])
        print(f"{ln['text']!r}: {len(ln['strokes'])} strokes, {n} points, ok={ln['ok']}")
    preview(result, args.out)
    print(f"preview: {args.out}")
