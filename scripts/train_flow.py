"""Train the flow-matching text-to-ink student (scripts/flow_model.py) on a
corpus of shards (teacher samples or MathWriting).

Loss = flow matching (velocity MSE) + --smooth-weight x neighbour-difference
MSE + --length-weight x length cross-entropy + --char-weight x token-index
cross-entropy. The flow loss is not a likelihood, so it cannot be compared
with the Mamba student's ink_nll; compare the sample images instead.

The run directory, resuming, the W&B corpus artifact and the sample images
work as in scripts/train_student.py. Unlike it, training stops when the
validation flow loss rises to DIVERGED times its best value, without saving,
so checkpoint.pt stays the last good one.

--init-from starts a new run from another flow checkpoint's weights, with a
fresh optimizer. The ink statistics are this corpus's, unless it is in the
checkpoint's ink units (same offset mean and std, as a corpus merged by
scripts/mixed_corpus.py is): then the checkpoint's are kept, so that it
carries on from what it learned. Its architecture must match, and this
corpus's vocabulary must start with the checkpoint's: tokens added after it
start from fresh embeddings.

Usage: python scripts/train_flow.py --run-dir models/student/<name>
           [--corpus data/teacher_corpus] [--steps 50000] [--max-points 65536]
           [--init-from models/student/<other>/checkpoint.pt] [--sample-steps 8]
           [--wandb-project P [--wandb-entity E] [--artifact teacher_corpus:latest]]
"""
import argparse
import json
import math
import os
import random
import time

import torch
import wandb

from corpus_artifact import add_wandb_args, pull, shard_names
from flow_model import FlowStudent, ink_statistics
from ink_corpus import Corpus
from sample_student import teacher_vs_student
from train_student import lr_at, start_wandb_run, to_device

LOSSES = ("flow", "smooth", "length", "char_ce", "char_acc")
DIVERGED = 1.5  # times the best validation flow loss


def total_loss(losses, args):
    return (losses["flow"] + args.smooth_weight * losses["smooth"] + args.length_weight * losses["length"]
            + args.char_weight * losses["char_ce"])


@torch.no_grad()
def evaluate(model, corpus, max_points, device, limit=200):
    model.eval()
    totals = dict.fromkeys(LOSSES, 0.0)
    batches = corpus.batches(max_points, random.Random(0))[:limit]
    # The loss draws noise and flow times; fixing them makes evaluations comparable.
    generator = torch.Generator(device=device).manual_seed(0)
    for idx in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(to_device(corpus.batch(idx), device), generator)
        for k in LOSSES:
            totals[k] += losses[k].item()
    model.train()
    return {k: v / len(batches) for k, v in totals.items()}


def main(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if os.path.exists(ckpt_path) else None

    run, names = None, None
    if args.wandb_project:
        if ckpt is not None:
            args.artifact = ckpt["args"].get("artifact", args.artifact)
        run = start_wandb_run(args)
        if args.artifact != "none":
            artifact = pull(args.corpus, args.wandb_entity, args.wandb_project, args.artifact, run)
            args.artifact = artifact.name  # pinned, e.g. teacher_corpus:v3, so chained jobs see the same data
            run.config.update({"artifact": args.artifact}, allow_val_change=True)
            names = shard_names(artifact)
            print(f"corpus {artifact.name}", flush=True)
    train, val = Corpus.split(args.corpus, args.val_shards, names)
    print(f"train {len(train)} lines, val {len(val)} lines", flush=True)

    init = None
    if ckpt is not None:
        config = ckpt["config"]
    else:
        coord_std, typical_height = ink_statistics(train)
        if args.init_from:
            init = torch.load(args.init_from, map_location=device, weights_only=False)
            trained = init["config"]
            if (trained["offset_mu"], trained["offset_std"]) == (train.meta["mu"], train.meta["std"]):
                # Same ink units, so the checkpoint's scale still fits and the
                # ink it already writes reaches it unchanged.
                coord_std, typical_height = trained["coord_std"], trained["typical_height"]
        config = dict(vocab_size=len(train.charset) + 2, offset_mu=train.meta["mu"], offset_std=train.meta["std"],
                      coord_std=coord_std, typical_height=typical_height, d_model=args.d_model,
                      n_layers=args.n_layers, heads=args.heads, text_layers=args.text_layers,
                      text_dropout=args.text_dropout, sample_steps=args.sample_steps)
    if run:
        run.config.update({"model_config": config}, allow_val_change=True)
    model = FlowStudent(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step, epoch, pos, best_val = 0, 0, 0, math.inf
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, epoch, pos, best_val = ckpt["step"], ckpt["epoch"], ckpt["pos"], ckpt["best_val"]
        print(f"resumed at step {step}", flush=True)
    elif init is not None:
        known = len(init["charset"])
        if train.charset[:known] != init["charset"]:
            raise ValueError(f"{args.init_from} was trained on a vocabulary this corpus's does not start with")
        # The buffers hold the ink statistics, which are this run's config's.
        state = {**init["model"], **dict(model.named_buffers())}
        # Tokens the checkpoint never saw keep their fresh embeddings.
        state["char_embed.weight"] = torch.cat([init["model"]["char_embed.weight"],
                                                model.char_embed.weight[known + 2:].detach()])
        model.load_state_dict(state)
        print(f"initialized from {args.init_from} at step {init['step']}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters; {model.config}", flush=True)

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"architecture": "flow", "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "epoch": epoch, "pos": pos, "best_val": best_val, "config": model.config, "charset": train.charset,
                    "meta": train.meta, "args": vars(args)}, tmp)
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
                losses = model.loss(to_device(train.batch(idx), device))
            loss = total_loss(losses, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite loss or gradient at step {step}")
            opt.step()
            step += 1
            pos += 1

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
                img = teacher_vs_student(model, val, args.sample_lines)
                img.save(os.path.join(args.run_dir, "samples", f"step_{step:07d}.png"))
                if run:
                    run.log({"samples": wandb.Image(img, caption="rows alternate corpus ink, flow student")},
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
    ap.add_argument("--corpus", default="data/teacher_corpus")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--max-points", type=int, default=65536, help="padded ink points per batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--init-from", help="flow checkpoint whose weights start a new run")
    ap.add_argument("--smooth-weight", type=float, default=1.0)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--char-weight", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--text-layers", type=int, default=3)
    ap.add_argument("--text-dropout", type=float, default=0.1,
                    help="fraction of lines trained without their text, for classifier-free guidance")
    ap.add_argument("--sample-steps", type=int, default=8,
                    help="Euler steps the model is sampled with, here and wherever its checkpoint is loaded")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--sample-lines", type=int, default=8, help="held-out lines drawn per sample image")
    ap.add_argument("--seed", type=int, default=0)
    add_wandb_args(ap, artifact_default="teacher_corpus:latest",
                   artifact_help="corpus artifact name, optionally :version; none uses --corpus as it is")
    main(ap.parse_args())
