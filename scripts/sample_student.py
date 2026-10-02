"""Sample the Mamba student: preview given lines, or measure how often it fails.

Preview: python scripts/sample_student.py --checkpoint models/student/<run>/checkpoint.pt
             "Some text" ["More lines" ...] [--bias 1.0] [--out preview.png]

Eval:    python scripts/sample_student.py --checkpoint ... --eval 1000
  Writes --eval random lines from the corpus line list and reports, by line
  length, how often the student fails: it never signals "finished", or it
  finishes with ink that text_to_ink.plausible() rejects. The teacher's
  rejection rate over the same line list is printed beside it. This is a
  proxy, not a quality measure: the teacher misspells often, and misspellings
  pass both checks. With --wandb-project, the corpus is pulled from W&B
  first (by default the version the checkpoint was trained on).
"""
import argparse
import glob
import os
import random

import numpy as np
import torch

from corpus_artifact import add_wandb_args, pull
from ink_corpus import char_tokens, encode_texts
from student_model import load_student
from text_to_ink import TEXT_HEIGHT, plausible, preview, to_line

# Rough teacher text height in IAM-OnDB units (descender line to cap line),
# from eyeballed samples. The student has no priming line to measure it from.
TEACHER_TEXT_HEIGHT = 500.0


class StudentWriter:
    """Same output as text_to_ink.Writer.write(), from the student."""

    def __init__(self, checkpoint, device="cuda"):
        self.model, ckpt = load_student(checkpoint, device)
        self.meta, self.corpus_dir = ckpt["meta"], ckpt["args"]["corpus"]
        self.artifact = ckpt["args"].get("artifact", "teacher_corpus:latest")
        self.mu, self.std = np.array(self.meta["mu"]), np.array(self.meta["std"])
        self.token = char_tokens(ckpt["charset"])
        self.device = device

    def write(self, lines, bias=1.0, seed=0, batch=128):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = []
        for b in range(0, len(lines), batch):
            chunk = lines[b:b + batch]
            text = encode_texts(chunk, self.token).to(self.device)
            text_len = torch.tensor([len(t) for t in chunk], device=self.device)
            for line, (off, chars, finished) in zip(chunk, self.model.sample(text, text_len, bias, generator=gen)):
                off = off.numpy()
                ok = finished and plausible(line, off * self.std + self.mu)
                out.append(to_line(line, off, chars.numpy(), self.mu, self.std, TEXT_HEIGHT / TEACHER_TEXT_HEIGHT, ok))
        return out


def teacher_rejection_rate(corpus_dir):
    kept = rejected = 0
    for path in glob.glob(os.path.join(corpus_dir, "shard_*.npz")):
        d = np.load(path)
        kept += len(d["texts"])
        rejected += int(d["rejected"])
    return rejected / max(1, kept + rejected)


def evaluate(writer, n, bias, seed):
    lines = [ln for ln in open(os.path.join(writer.corpus_dir, writer.meta["lines"])).read().split("\n") if ln]
    lines = sorted(random.Random(seed).sample(lines, n), key=len)
    result = writer.write(lines, bias, seed)
    buckets = [(1, 10), (11, 25), (26, 50)]
    for lo, hi in buckets:
        rows = [r for r in result if lo <= len(r["text"]) <= hi]
        if rows:
            fail = sum(not r["ok"] for r in rows) / len(rows)
            print(f"length {lo:2d}-{hi:2d}: {len(rows):5d} lines, student fails {fail:.1%}")
    print(f"all: student fails {sum(not r['ok'] for r in result) / len(result):.1%}, "
          f"teacher rejected {teacher_rejection_rate(writer.corpus_dir):.1%} of its attempts")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lines", nargs="*")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bias", type=float, default=1.0, help="0 = most varied, higher = neater")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=int, default=0, help="number of random corpus lines to evaluate on")
    ap.add_argument("--out", default="student_preview.png")
    add_wandb_args(ap, artifact_default=None,
                   artifact_help="corpus artifact for --eval (default: the version the checkpoint was trained on)")
    args = ap.parse_args()

    writer = StudentWriter(args.checkpoint)
    if args.eval and args.wandb_project:
        pull(writer.corpus_dir, args.wandb_entity, args.wandb_project, args.artifact or writer.artifact)
    if args.eval:
        result = evaluate(writer, args.eval, args.bias, args.seed)
        preview(random.Random(args.seed).sample(result, min(20, len(result))), args.out)
    else:
        result = writer.write(args.lines, args.bias, args.seed)
        for r in result:
            print(f"{r['text']!r}: {len(r['strokes'])} strokes, ok={r['ok']}")
        preview(result, args.out)
    print(f"preview: {args.out}")
