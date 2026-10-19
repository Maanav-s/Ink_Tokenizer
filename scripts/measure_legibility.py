"""Measure how legible a student's ink is: the student writes held-out labels,
the recognizer (scripts/train_recognizer.py) reads them back, and the token
error rate (edit distance / label tokens) and exact-read rate are reported.

The recognizer also reads the real ink of the same labels, so the student's
numbers can be set against how well it reads human handwriting. With
--samples N > 1, each label is written N times: the rates are averaged over
samples, and "best of N" is the fraction of labels with at least one sample
read exactly (what picking samples with the recognizer would give).

The most common token mistakes in the student's ink are listed, and its --n
worst-read samples are drawn as scripts/inspect_recognizer.py draws them.
With --wandb-project, everything also goes to a new W&B run (job type
"legibility").

Usage: python scripts/measure_legibility.py --student models/student/<run>/checkpoint.pt
           --recognizer models/recognizer/<run>/checkpoint.pt [--corpus DIR] [--lines 1000]
           [--samples 1] [--steps S] [--guidance G] [--bias 0] [--batch 16] [--out PATH]
           [--wandb-project P [--wandb-entity E]]
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import torch

from ink_corpus import Corpus, char_tokens, encode_texts, split_text
from inspect_recognizer import draw, edit_ops
from recognizer_model import Recognizer, ink_positions
from sample_student import load_model
from train_recognizer import inputs, to_device, unwrap

BATCH = 64


def pad(offsets):
    """A list of (n, 3) offset tensors -> (B, N, 3) zero-padded, and the lengths."""
    lengths = torch.tensor([len(o) for o in offsets])
    out = torch.zeros(len(offsets), int(lengths.max()), 3)
    for b, o in enumerate(offsets):
        out[b, :len(o)] = o
    return out, lengths


def read_inks(recognizer, ink, ink_len):
    """Token ids the recognizer reads in each ink."""
    reads = []
    for b in range(0, len(ink), BATCH):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reads += recognizer.read(ink[b:b + BATCH], ink_len[b:b + BATCH])
    return reads


def score(reads, targets, charset):
    """Per ink: (read tokens, edit ops)."""
    out = []
    for ids, target in zip(reads, targets):
        read = [charset[t - 2] for t in ids]
        out.append((read, edit_ops(read, target)))
    return out


def rates(scored, targets):
    errors = sum(len(ops) for _, ops in scored)
    tokens = sum(len(t) for t in targets)
    return errors / tokens, sum(not ops for _, ops in scored) / len(scored)


@torch.no_grad()
def main(args):
    device = "cuda"
    student, student_ckpt = load_model(args.student, device)
    if args.steps is not None:
        student.sample_steps = args.steps
    if args.guidance is not None:
        student.sample_guidance = args.guidance
    rec_ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    recognizer = Recognizer(**rec_ckpt["config"]).to(device)
    recognizer.load_state_dict(rec_ckpt["model"])
    recognizer.eval()

    corpus_dir = args.corpus or student_ckpt["args"]["corpus"]
    shards = sorted(glob.glob(os.path.join(corpus_dir, "shard_*.npz")))
    val = Corpus(corpus_dir, shards[-student_ckpt["args"].get("val_shards", 1):])
    val.texts = [unwrap(t) for t in val.texts]
    student_meta, tokenizer = student_ckpt["meta"], student_ckpt["meta"].get("tokenizer", "char")
    student_token = char_tokens(student_ckpt["charset"])
    readable = set(rec_ckpt["charset"])
    candidates = [i for i, t in enumerate(val.texts)
                  if set(split_text(t, tokenizer)) <= set(student_token) and set(split_text(t, tokenizer)) <= readable]
    picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(args.lines, len(candidates))).astype(int)]
    labels = [val.texts[i] for i in picks]
    targets = [split_text(t, tokenizer) for t in labels]
    print(f"{len(picks)} labels from {corpus_dir} ({len(candidates)} of {len(val)} in both vocabularies)", flush=True)

    # The real ink of the same labels: how well the recognizer reads human handwriting.
    mu = torch.tensor(val.meta["mu"], dtype=torch.float32, device=device)
    std = torch.tensor(val.meta["std"], dtype=torch.float32, device=device)
    real_reads = []
    for b in range(0, len(picks), BATCH):
        batch = to_device(val.batch(picks[b:b + BATCH]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            real_reads += recognizer.read(*inputs(batch, mu, std))
    real = score(real_reads, targets, rec_ckpt["charset"])
    real_ter, real_exact = rates(real, targets)

    student_mu = torch.tensor(student_meta["mu"], dtype=torch.float32, device=device)
    student_std = torch.tensor(student_meta["std"], dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    samples = []  # per sample round: list of (scored, ink, ink_len) per label
    for _ in range(args.samples):
        offsets = []
        for b in range(0, len(labels), args.batch):
            chunk = labels[b:b + args.batch]
            text, text_len = encode_texts(chunk, student_token, tokenizer)
            bias = torch.full((len(chunk), 1, 1), args.bias, device=device)
            offsets += [off for off, _, _ in student.sample(text.to(device), text_len.to(device), bias, generator=gen)]
        off, ink_len = pad(offsets)
        ink = ink_positions(off.to(device), ink_len.to(device), student_mu, student_std)
        reads = read_inks(recognizer, ink, ink_len.to(device))
        samples.append((score(reads, targets, rec_ckpt["charset"]), ink.cpu(), ink_len))

    per_round = [rates(scored, targets) for scored, _, _ in samples]
    ter = float(np.mean([r[0] for r in per_round]))
    exact = float(np.mean([r[1] for r in per_round]))
    best_of_n = float(np.mean([any(not samples[s][0][i][1] for s in range(args.samples)) for i in range(len(labels))]))
    result = {"student_token_error_rate": ter, "student_exact": exact, "real_token_error_rate": real_ter,
              "real_exact": real_exact}
    if args.samples > 1:
        result[f"student_best_of_{args.samples}"] = best_of_n
    print(f"student: token error rate {ter:.3f}, read exactly {exact:.1%}"
          + (f", best of {args.samples} {best_of_n:.1%}" if args.samples > 1 else ""))
    print(f"real ink of the same labels: token error rate {real_ter:.3f}, read exactly {real_exact:.1%}")

    buckets = [(1, 5), (6, 10), (11, 20), (21, 50)]
    for lo, hi in buckets:
        idx = [i for i, t in enumerate(targets) if lo <= len(t) <= hi]
        if idx:
            sub = [targets[i] for i in idx]
            s_ter, s_exact = rates([samples[0][0][i] for i in idx], sub)
            r_ter, r_exact = rates([real[i] for i in idx], sub)
            print(f"  {lo:2d}-{hi:2d} tokens, {len(idx):4d} labels: student {s_ter:.3f} / {s_exact:.1%}, "
                  f"real {r_ter:.3f} / {r_exact:.1%}")

    mistakes = Counter(op for scored, _, _ in samples for _, ops in scored for op in ops)
    print("most common mistakes in the student's ink (sub: label -> read; drop: in the label, not read; "
          "add: read, not in the label):")
    for op, count in mistakes.most_common(args.top):
        print(f"  {count:5d}  {op[0]:4s} " + " -> ".join(op[1:]))

    scored, ink, ink_len = samples[0]
    worst = sorted(range(len(labels)), key=lambda i: -len(scored[i][1]) / len(targets[i]))[:args.n]
    rows = [(ink[i].float().numpy(), int(ink_len[i]), " ".join(targets[i]), " ".join(scored[i][0]))
            for i in worst if scored[i][1]]
    name = os.path.basename(os.path.dirname(os.path.abspath(args.student)))
    image = None
    if rows:
        out = args.out or os.path.join("artifacts", "recognizer", f"{name}_legibility.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        image = draw(rows)
        image.save(out)
        print(f"drew the {len(rows)} worst-read samples: {out}")

    if args.wandb_project:
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="legibility",
                         name=f"{name}-legibility",
                         config={**vars(args), "student_step": student_ckpt["step"],
                                 "recognizer_step": rec_ckpt["step"], "corpus": corpus_dir,
                                 "sample_steps": getattr(student, "sample_steps", None),
                                 "sample_guidance": getattr(student, "sample_guidance", None)})
        mistake_rows = [[op[0], op[1], op[2] if op[0] == "sub" else "", count] if op[0] != "add"
                        else [op[0], "", op[1], count] for op, count in mistakes.most_common()]
        reads = [[" ".join(t), " ".join(s[0]), len(s[1]), " ".join(r[0]), len(r[1])]
                 for t, s, r in zip(targets, scored, real)]
        record = {**result,
                  "mistakes": wandb.Table(columns=["kind", "label token", "read token", "count"], data=mistake_rows),
                  "reads": wandb.Table(columns=["label", "student read", "student errors", "real read",
                                                "real errors"], data=reads)}
        if image is not None:
            record["worst"] = wandb.Image(image, caption=f"{name}: worst-read student samples, label green")
        run.log(record)
        run.summary.update(result)
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--corpus", help="corpus whose validation labels are written (default: the student's)")
    ap.add_argument("--lines", type=int, default=1000, help="labels written, spread over the validation shard")
    ap.add_argument("--samples", type=int, default=1, help="samples per label")
    ap.add_argument("--steps", type=int, help="Euler steps (flow students; default: the checkpoint's)")
    ap.add_argument("--guidance", type=float, help="guidance scale (flow students; default: the checkpoint's)")
    ap.add_argument("--bias", type=float, default=0.0, help="sampling bias; MathWriting corpora are written at 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16,
                    help="labels the student writes at once; its attention over ~2000 points grows with the square")
    ap.add_argument("--n", type=int, default=24, help="worst-read samples drawn")
    ap.add_argument("--top", type=int, default=25, help="mistakes listed")
    ap.add_argument("--out", help="image path (default: artifacts/recognizer/<student run>_legibility.png)")
    ap.add_argument("--wandb-entity", default=None, help="W&B user or team (default: your W&B default entity)")
    ap.add_argument("--wandb-project", default=None, help="W&B project; W&B is not used without it")
    main(ap.parse_args())
