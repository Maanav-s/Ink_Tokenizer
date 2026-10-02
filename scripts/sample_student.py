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
from text_to_ink import TEXT_HEIGHT, ink_extent, plausible, preview, render, to_line

# Rough teacher text height in IAM-OnDB units (descender line to cap line),
# from eyeballed samples. The student has no priming line to measure it from.
TEACHER_TEXT_HEIGHT = 500.0


def preview_scale(meta):
    """mm per denormalized ink unit, for drawing a corpus's ink."""
    return TEXT_HEIGHT / meta.get("text_height", TEACHER_TEXT_HEIGHT)


def write_student(model, token, meta, lines, bias=1.0, seed=0, batch=128):
    """to_line() dicts for lines written by the student. bias is one value or
    one per line. plausible() is tuned to the teacher's ink, so it is only
    applied to students of a teacher corpus."""
    mu, std = np.array(meta["mu"]), np.array(meta["std"])
    scale, teacher = preview_scale(meta), "teacher" in meta
    device = next(model.parameters()).device
    bias = np.broadcast_to(np.asarray(bias, dtype=np.float32), (len(lines),))
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for b in range(0, len(lines), batch):
        chunk = lines[b:b + batch]
        text, text_len = encode_texts(chunk, token, meta.get("tokenizer", "char"))
        text, text_len = text.to(device), text_len.to(device)
        line_bias = torch.from_numpy(bias[b:b + batch].copy()).to(device).view(-1, 1, 1)
        for line, (off, chars, finished) in zip(chunk, model.sample(text, text_len, line_bias, generator=gen)):
            off = off.numpy()
            ok = finished and (not teacher or plausible(line, off * std + mu))
            out.append(to_line(line, off, chars.numpy(), mu, std, scale, ok))
    return out


@torch.no_grad()
def teacher_vs_student(model, corpus, n=8, seed=0):
    """An image whose rows alternate between the corpus ink for a line (the
    teacher's, or real ink) and the student writing the same line at the
    same bias. The n lines
    are spread over the corpus's line lengths. Dimmed student rows failed.
    Student ink far outside the teacher's extent is cut off."""
    order = np.argsort([len(t) for t in corpus.texts], kind="stable")
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    texts = [corpus.texts[i] for i in picks]
    mu, std = np.array(corpus.meta["mu"]), np.array(corpus.meta["std"])
    was_training = model.training
    model.eval()
    student = write_student(model, corpus.token, corpus.meta, texts, corpus.bias[picks], seed)
    model.train(was_training)
    rows = []
    for i, text, s in zip(picks, texts, student):
        a, n_points = corpus.starts[i], corpus.lengths[i]
        rows.append(to_line(text, corpus.offsets[a:a + n_points], corpus.chars[a:a + n_points], mu, std,
                            preview_scale(corpus.meta)))
        rows.append(s)
    teacher = [ink_extent(r) for r in rows[::2]]
    return render(rows, max_width=1.5 * max(e[0] for e in teacher),
                  max_height=1.5 * max(max(-e[1], e[2]) for e in teacher))


class StudentWriter:
    """Same output as text_to_ink.Writer.write(), from the student."""

    def __init__(self, checkpoint, device="cuda"):
        self.model, ckpt = load_student(checkpoint, device)
        self.meta, self.corpus_dir = ckpt["meta"], ckpt["args"]["corpus"]
        self.artifact = ckpt["args"].get("artifact", "teacher_corpus:latest")
        self.token = char_tokens(ckpt["charset"])
        self.device = device

    def write(self, lines, bias=1.0, seed=0, batch=128):
        return write_student(self.model, self.token, self.meta, lines, bias, seed, batch)


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
    artifact = args.artifact or writer.artifact
    if args.eval and args.wandb_project and artifact != "none":
        pull(writer.corpus_dir, args.wandb_entity, args.wandb_project, artifact)
    if args.eval:
        result = evaluate(writer, args.eval, args.bias, args.seed)
        preview(random.Random(args.seed).sample(result, min(20, len(result))), args.out)
    else:
        result = writer.write(args.lines, args.bias, args.seed)
        for r in result:
            print(f"{r['text']!r}: {len(r['strokes'])} strokes, ok={r['ok']}")
        preview(result, args.out)
    print(f"preview: {args.out}")
