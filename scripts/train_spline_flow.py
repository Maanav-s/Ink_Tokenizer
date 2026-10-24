"""Train the spline flow student (scripts/spline_flow_model.py) on a corpus
converted by scripts/spline_corpus.py.

The losses, run directory, resuming, W&B and the divergence stop work as in
scripts/train_flow.py, but its losses are over control points, so they are
not comparable with the point flow student's; compare the sample images, or
the recognizer's reads of samples. Sample images draw the source corpus's
held-out ink (meta.json "spline" source, which must be present) next to the
student's decoded samples.

Not carried over from train_flow.py, to keep the experiment small:
--init-from (a point checkpoint's length head and positions mean something
else here) and --stretch-prob (resample_ink interpolates control points,
which changes the curve).

Usage: python scripts/train_spline_flow.py --run-dir models/student/<name>
           [--corpus data/spline_corpus] [--steps 50000] [--max-points 32768] [--accumulate 2]
           [--wandb-project P [--wandb-entity E]]
"""
import argparse
import json
import math
import os
import random
import time

import torch
import wandb

from corpus_artifact import add_wandb_args
from flow_model import ink_statistics
from ink_corpus import Corpus
from sample_student import teacher_vs_student
from spline_flow_model import SplineFlowStudent
from train_flow import DIVERGED, LOSSES, evaluate, total_loss
from train_student import lr_at, start_wandb_run, to_device


def main(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if os.path.exists(ckpt_path) else None

    run = start_wandb_run(args) if args.wandb_project else None
    train, val = Corpus.split(args.corpus, args.val_shards)
    if "spline" not in train.meta:
        raise ValueError(f"{args.corpus} is not a spline corpus (scripts/spline_corpus.py)")
    _, source_val = Corpus.split(train.meta["spline"]["source"], args.val_shards)
    print(f"train {len(train)} lines, val {len(val)} lines", flush=True)

    if ckpt is not None:
        config = ckpt["config"]
    else:
        coord_std, typical_height = ink_statistics(train)
        config = dict(points_per_control=train.meta["spline"]["points_per_control"],
                      vocab_size=len(train.charset) + 2, offset_mu=train.meta["mu"], offset_std=train.meta["std"],
                      coord_std=coord_std, typical_height=typical_height, d_model=args.d_model,
                      n_layers=args.n_layers, heads=args.heads, text_layers=args.text_layers,
                      text_dropout=args.text_dropout, sample_steps=args.sample_steps,
                      sample_length=args.sample_length)
    if run:
        run.config.update({"model_config": config}, allow_val_change=True)
    model = SplineFlowStudent(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step, epoch, pos, best_val = 0, 0, 0, math.inf
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, epoch, pos, best_val = ckpt["step"], ckpt["epoch"], ckpt["pos"], ckpt["best_val"]
        print(f"resumed at step {step}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters; {model.config}", flush=True)

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"architecture": "spline_flow", "model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "epoch": epoch, "pos": pos, "best_val": best_val, "config": model.config,
                    "charset": train.charset, "meta": train.meta, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)

    log = open(os.path.join(args.run_dir, "metrics.jsonl"), "a")
    model.train()
    start = time.time()
    while step < args.steps:
        batches = train.batches(args.max_points, random.Random(args.seed * 1000 + epoch))
        while pos < len(batches) and step < args.steps:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
            opt.zero_grad(set_to_none=True)
            micro = batches[pos:pos + args.accumulate]
            losses = dict.fromkeys(LOSSES, 0.0)
            for idx in micro:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    batch_losses = model.loss(to_device(train.batch(idx), device))
                loss = total_loss(batch_losses, args)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                (loss / len(micro)).backward()
                for k in LOSSES:
                    losses[k] += batch_losses[k].detach() / len(micro)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {step}")
            opt.step()
            step += 1
            pos += len(micro)

            if step % args.log_every == 0 or step % args.eval_every == 0:
                rec = {"step": step, "epoch": epoch, **{k: losses[k].item() for k in LOSSES},
                       "grad_norm": grad_norm.item(), "lr": opt.param_groups[0]["lr"],
                       "elapsed_s": round(time.time() - start)}
                diverged = False
                if step % args.eval_every == 0:
                    v = evaluate(model, val, args.max_points, device)
                    rec.update({f"val_{k}": x for k, x in v.items()})
                    diverged = v["flow"] > DIVERGED * best_val
                    if not diverged:
                        best_val = min(best_val, v["flow"])
                        save()
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if run:
                    run.log(rec, step=step)
                if diverged:
                    raise RuntimeError(f"diverged at step {step}: val flow {v['flow']:.3f}, best {best_val:.3f}; "
                                       f"checkpoint.pt is from the evaluation before")
            if step % args.sample_every == 0:
                os.makedirs(os.path.join(args.run_dir, "samples"), exist_ok=True)
                img = teacher_vs_student(model, source_val, args.sample_lines)
                img.save(os.path.join(args.run_dir, "samples", f"step_{step:07d}.png"))
                if run:
                    run.log({"samples": wandb.Image(img, caption="rows alternate source ink, spline flow student")},
                            step=step)
        if pos >= len(batches):
            epoch, pos = epoch + 1, 0
    save()
    print(f"done at step {step}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--corpus", default="data/spline_corpus")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--max-points", type=int, default=32768,
                    help="padded control points per batch; each stands for about 3 points")
    ap.add_argument("--accumulate", type=int, default=1, help="batches whose gradients are summed per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=1.0)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--char-weight", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--text-layers", type=int, default=3)
    ap.add_argument("--text-dropout", type=float, default=0.1,
                    help="fraction of lines trained without their text, for classifier-free guidance")
    ap.add_argument("--sample-steps", type=int, default=8)
    ap.add_argument("--sample-length", choices=("argmax", "sample"), default="argmax")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--sample-lines", type=int, default=8, help="held-out lines drawn per sample image")
    ap.add_argument("--seed", type=int, default=0)
    add_wandb_args(ap, artifact_default="none", artifact_help="unused: spline corpora are converted locally")
    main(ap.parse_args())
