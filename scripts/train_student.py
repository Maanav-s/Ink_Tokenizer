"""Train the Mamba text-to-ink student (scripts/student_model.py) on a teacher
corpus (scripts/generate_teacher_corpus.py).

Loss = ink NLL (per point, the teacher's mixture form) + --char-weight x
character-index cross-entropy. The run directory holds checkpoint.pt (latest,
with optimizer state) and metrics.jsonl. Rerunning with the same --run-dir
resumes, so a job that hits its wall time can be chained with another.

Usage: python scripts/train_student.py --run-dir models/student/<name>
           [--corpus data/teacher_corpus] [--steps 50000] [--max-points 65536]
"""
import argparse
import json
import math
import os
import random
import time

import torch

from ink_corpus import Corpus
from student_model import Student


def lr_at(step, peak, warmup, total):
    if step < warmup:
        return peak * step / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, corpus, max_points, device, limit=200):
    model.eval()
    totals = [0.0, 0.0, 0.0]
    batches = corpus.batches(max_points, random.Random(0))[:limit]
    for idx in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i, v in enumerate(model.loss(to_device(corpus.batch(idx), device))):
                totals[i] += v.item()
    model.train()
    return [t / len(batches) for t in totals]


def main(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    train, val = Corpus.split(args.corpus, args.val_shards)
    print(f"train {len(train)} lines, val {len(val)} lines", flush=True)

    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    config = dict(vocab_size=len(train.charset) + 2, d_model=args.d_model, n_layers=args.n_layers,
                  d_state=args.d_state, headdim=args.headdim)
    model = Student(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step, epoch, pos = 0, 0, 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, epoch, pos = ckpt["step"], ckpt["epoch"], ckpt["pos"]
        print(f"resumed at step {step}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                    "pos": pos, "config": model.config, "charset": train.charset, "meta": train.meta,
                    "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)

    log = open(os.path.join(args.run_dir, "metrics.jsonl"), "a")
    model.train()
    start = time.time()
    while step < args.steps:
        # Each epoch's batch order depends only on the seed and epoch, so a
        # resumed run continues where it stopped instead of replaying data.
        batches = train.batches(args.max_points, random.Random(args.seed * 1000 + epoch))
        while pos < len(batches) and step < args.steps:
            idx = batches[pos]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ink_nll, char_ce, char_acc = model.loss(to_device(train.batch(idx), device))
                loss = ink_nll + args.char_weight * char_ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.step()
            step += 1
            pos += 1

            if step % args.log_every == 0 or step % args.eval_every == 0:
                rec = {"step": step, "epoch": epoch, "ink_nll": ink_nll.item(), "char_ce": char_ce.item(),
                       "char_acc": char_acc.item(), "grad_norm": grad_norm.item(),
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - start)}
                if step % args.eval_every == 0:
                    v = evaluate(model, val, args.max_points, device)
                    rec.update(val_ink_nll=v[0], val_char_ce=v[1], val_char_acc=v[2])
                    save()
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if pos >= len(batches):
            epoch, pos = epoch + 1, 0
    save()
    print(f"done at step {step}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--corpus", default="data/teacher_corpus")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--max-points", type=int, default=65536, help="padded ink points per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--char-weight", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--d-state", type=int, default=64)
    ap.add_argument("--headdim", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
