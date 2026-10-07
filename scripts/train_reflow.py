"""Reflow a trained flow student (scripts/flow_model.py) so that it writes in
fewer Euler steps.

The teacher is a flow checkpoint. For every batch of corpus lines it carries
fresh noise to ink with --teacher-steps guided Euler steps, and the student,
which starts from the teacher's weights, is trained with the flow-matching
loss on that (noise, ink) pair instead of on corpus ink paired with unrelated
noise. Each noise now has one ink it belongs to, so the straight path between
them is nearly the path the student should follow, and a few large steps
land close to where many small ones do.

The teacher's guidance is part of the pairs. The student is therefore trained
with its text always present and sampled without guidance (one pass per step,
not two); its checkpoint records --student-steps and guidance 1 as its
sampling defaults, which scripts/sample_student.py and the sample images use.

Only the corpus's text and point counts are used: the ink is the teacher's,
and so are the point labels where the corpus has an alignment. The noise temperature follows a bias drawn per line
from [0, --bias-max], so the student covers the biases it is sampled at.

A training step costs about as much as --teacher-steps x 2 extra forward
passes. Losses are measured against the teacher's ink, so they are not
comparable with train_flow.py's; the sample images are the thing to watch.
Each draws held-out lines as corpus ink and then as the student writes them
at every step count in --show-steps, from the same noise. The run directory, resuming, W&B and the
divergence stop work as in train_flow.py; the corpus and its artifact are
the teacher's unless given.

Usage: python scripts/train_reflow.py --run-dir models/student/<name>
           --teacher models/student/<flow run>/checkpoint.pt
           [--student-steps 4] [--teacher-steps 32] [--guidance 2.0] [--steps 20000]
           [--wandb-project P [--wandb-entity E]]
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import wandb

from corpus_artifact import add_wandb_args, pull, shard_names
from flow_model import BIAS_TEMPERATURE, FlowStudent
from ink_corpus import Corpus
from sample_student import load_model, preview_scale, write_student
from text_to_ink import ink_extent, render, to_line
from train_flow import DIVERGED, LOSSES, total_loss
from train_student import lr_at, start_wandb_run, to_device


@torch.no_grad()
def reflow_batch(teacher, batch, args, generator=None):
    """The batch with the teacher's point labels in place of the corpus's,
    and the (noise, ink) pair the teacher made for it."""
    B, N, _ = batch["offsets"].shape
    device = batch["offsets"].device
    bias = torch.rand(B, generator=generator, device=device) * args.bias_max
    noise = torch.randn((B, N, 3), generator=generator, device=device)
    noise = noise * torch.exp(-BIAS_TEMPERATURE * bias).view(B, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ink, chars = teacher.transport(noise, batch["text"], batch["text_len"], batch["ink_len"],
                                       args.teacher_steps, args.guidance)
    # Without an alignment in the corpus the teacher's label head is untrained.
    chars = torch.where(batch["chars"] >= 0, chars, -1)
    return {**batch, "chars": chars}, (noise, ink)


@torch.no_grad()
def evaluate(model, teacher, corpus, args, device):
    model.eval()
    totals = dict.fromkeys(LOSSES, 0.0)
    batches = corpus.batches(args.max_points, random.Random(0))[:args.eval_batches]
    # Fixed noise and flow times give every evaluation the same teacher pairs.
    generator = torch.Generator(device=device).manual_seed(0)
    for idx in batches:
        batch, pair = reflow_batch(teacher, to_device(corpus.batch(idx), device), args, generator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(batch, generator, pair)
        for k in LOSSES:
            totals[k] += losses[k].item()
    model.train()
    return {k: v / len(batches) for k, v in totals.items()}


@torch.no_grad()
def step_count_image(model, corpus, n, step_counts):
    """An image of n corpus lines spread over the line lengths. Each line is
    drawn as corpus ink and then as the student writes it at each of
    step_counts, top to bottom, from the same noise and point count. Student
    ink far outside the corpus ink's extent is cut off."""
    order = np.argsort([len(t) for t in corpus.texts], kind="stable")
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    texts = [corpus.texts[i] for i in picks]
    mu, std = np.array(corpus.meta["mu"]), np.array(corpus.meta["std"])
    model.eval()
    written = []
    for steps in step_counts:
        model.sample_steps = steps
        written.append(write_student(model, corpus.token, corpus.meta, texts, corpus.bias[picks]))
    model.sample_steps = model.config["sample_steps"]
    model.train()
    rows, reference = [], []
    for line, (i, text) in enumerate(zip(picks, texts)):
        a, n_points = corpus.starts[i], corpus.lengths[i]
        rows.append(to_line(text, corpus.offsets[a:a + n_points], corpus.chars[a:a + n_points], mu, std,
                            preview_scale(corpus.meta)))
        reference.append(ink_extent(rows[-1]))
        rows.extend(w[line] for w in written)
    return render(rows, max_width=1.5 * max(e[0] for e in reference),
                  max_height=1.5 * max(max(-e[1], e[2]) for e in reference))


def main(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if os.path.exists(ckpt_path) else None

    teacher, teacher_ckpt = load_model(args.teacher, device)
    if teacher_ckpt.get("architecture") != "flow":
        raise ValueError(f"{args.teacher} is not a flow checkpoint")
    teacher.requires_grad_(False)
    args.corpus = args.corpus or teacher_ckpt["args"]["corpus"]
    args.artifact = args.artifact or teacher_ckpt["args"].get("artifact", "none")
    print(f"teacher {args.teacher} at step {teacher_ckpt['step']}", flush=True)

    run, names = None, None
    if args.wandb_project:
        run = start_wandb_run(args)
        if args.artifact != "none":
            artifact = pull(args.corpus, args.wandb_entity, args.wandb_project, args.artifact, run)
            names = shard_names(artifact)
            print(f"corpus {artifact.name}", flush=True)
    train, val = Corpus.split(args.corpus, args.val_shards, names)
    print(f"train {len(train)} lines, val {len(val)} lines", flush=True)

    if ckpt is not None:
        config = ckpt["config"]
    else:
        config = {**teacher.config, "text_dropout": 0.0, "sample_steps": args.student_steps, "sample_guidance": 1.0}
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
    else:
        model.load_state_dict(teacher.state_dict())
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
            batch, pair = reflow_batch(teacher, to_device(train.batch(idx), device), args)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model.loss(batch, pair=pair)
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
                    v = evaluate(model, teacher, val, args, device)
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
                img = step_count_image(model, val, args.sample_lines, args.show_steps)
                img.save(os.path.join(args.run_dir, "samples", f"step_{step:07d}.png"))
                if run:
                    caption = f"per line: corpus ink, then the reflowed student at {args.show_steps} steps"
                    run.log({"samples": wandb.Image(img, caption=caption)}, step=step)
        if pos >= len(batches):
            epoch, pos = epoch + 1, 0
    save()
    print(f"done at step {step}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--teacher", required=True, help="flow checkpoint to reflow")
    ap.add_argument("--corpus", help="default: the teacher's")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--student-steps", type=int, default=4, help="Euler steps the student is sampled with")
    ap.add_argument("--teacher-steps", type=int, default=32, help="Euler steps the teacher makes each pair with")
    ap.add_argument("--guidance", type=float, default=2.0, help="the teacher's classifier-free guidance scale")
    ap.add_argument("--bias-max", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-points", type=int, default=65536, help="padded ink points per batch")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=1.0)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--char-weight", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--sample-lines", type=int, default=4, help="held-out lines drawn per sample image")
    ap.add_argument("--show-steps", type=int, nargs="+", default=[2, 4, 8, 16],
                    help="Euler step counts each of those lines is drawn at")
    ap.add_argument("--seed", type=int, default=0)
    add_wandb_args(ap, artifact_default=None,
                   artifact_help="corpus artifact name, optionally :version; none uses --corpus as it is "
                                 "(default: the teacher's)")
    main(ap.parse_args())
