r"""Fine-tune a flow student (scripts/flow_model.py) so that the frozen
recognizer (scripts/recognizer_model.py) reads its ink as the label it was
asked to write.

Loss = train_flow.py's losses, unchanged, + --legibility-weight x the
recognizer's token cross-entropy on the student's one-step estimate of the
finished ink, x̂1 = xt + (1 - t) v. That term is computed in a second pass
whose flow times are drawn uniformly from [--legibility-t-min, 1], so it only
applies once the ink is near its final form: earlier, x̂1 is a blur the
recognizer was never trained on, and reading it would push the student
towards whatever the recognizer happens to accept there. The flow pass keeps
its own logit-normal times, so the flow loss trains as before and stays
dominant; the student should not win by writing ink only this recognizer can
read, so judge the result with scripts/measure_legibility.py (ideally with a
second recognizer).

The recognizer's weights are frozen, but gradients flow through it into x̂1.
Lines it cannot read (a token outside its vocabulary, such as the teacher's
\text{...} lines in the mixed corpus) get no legibility term; `\typeset{...}`
lines are read as their label, as in train_recognizer.py.

--init-from is the student checkpoint, whose architecture and ink statistics
are kept; the corpus (by default the checkpoint's) must have its vocabulary.
The run directory, resuming, W&B, sample images and the divergence stop (on
the validation flow loss) work as in train_flow.py.

Usage: python scripts/train_flow_legibility.py --run-dir models/student/<name>
           --init-from models/student/<flow run>/checkpoint.pt
           --recognizer models/recognizer/<run>/checkpoint.pt
           [--corpus DIR] [--steps 10000] [--legibility-weight 0.1] [--legibility-t-min 0.8]
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

from flow_model import FlowStudent
from ink_corpus import Corpus, char_tokens, encode_texts, split_text
from recognizer_model import Recognizer
from sample_student import teacher_vs_student
from train_flow import DIVERGED, LOSSES, total_loss
from train_flow import evaluate as evaluate_flow
from train_recognizer import unwrap
from train_student import lr_at, start_wandb_run, to_device


def recognizer_labels(corpus, rec_token):
    """Per line, the label the recognizer should read, or None where it has no
    token for part of it."""
    out = []
    for text in corpus.texts:
        label = unwrap(text)
        out.append(label if set(split_text(label, "latex")) <= rec_token.keys() else None)
    return out


def legibility_loss(model, recognizer, batch, labels, rec_token, t_min, generator=None):
    """The recognizer's cross-entropy on x̂1 at flow times in [t_min, 1], over
    the rows with a label. None if no row has one."""
    keep = [b for b, label in enumerate(labels) if label is not None]
    if not keep:
        return None
    device = batch["offsets"].device
    rows = torch.tensor(keep, device=device)
    text, text_len, offsets, ink_len = (batch[k][rows] for k in ("text", "text_len", "offsets", "ink_len"))
    B, N, _ = offsets.shape
    mask = (torch.arange(N, device=device).unsqueeze(0) < ink_len.unsqueeze(1)).unsqueeze(-1)

    x1 = model.to_positions(offsets, ink_len)
    x0 = torch.randn(x1.shape, generator=generator, device=device)
    t = t_min + (1 - t_min) * torch.rand(B, generator=generator, device=device)
    xt = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
    # Always with the text: dropping it (for guidance) leaves nothing to read.
    memory, memory_pad, _ = model.encode_text(text, text_len)
    velocity, _ = model.denoise(xt, t, ink_len, memory, memory_pad)
    x1_hat = (xt + (1 - t.view(B, 1, 1)) * velocity) * mask

    ink = torch.cat([x1_hat[..., :2] * model.coord_std, x1_hat[..., 2:]], dim=-1)
    target, target_len = encode_texts([labels[b] for b in keep], rec_token, "latex")
    return recognizer.loss(ink, ink_len, target.to(device), target_len.to(device))["ce"]


@torch.no_grad()
def evaluate(model, recognizer, corpus, labels, rec_token, args, device, limit=200):
    out = evaluate_flow(model, corpus, args.max_points, device, limit)
    model.eval()
    batches = corpus.batches(args.max_points, random.Random(0))[:limit]
    generator = torch.Generator(device=device).manual_seed(0)
    total, counted = 0.0, 0
    for idx in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = legibility_loss(model, recognizer, to_device(corpus.batch(idx), device),
                                   [labels[i] for i in idx], rec_token, args.legibility_t_min, generator)
        if loss is not None:
            total, counted = total + loss.item(), counted + 1
    model.train()
    out["legibility"] = total / max(counted, 1)
    return out


def main(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if os.path.exists(ckpt_path) else None
    init = torch.load(args.init_from, map_location=device, weights_only=False)
    if init.get("architecture") != "flow":
        raise SystemExit(f"{args.init_from} is not a flow checkpoint")
    args.corpus = args.corpus or init["args"]["corpus"]

    run = start_wandb_run(args) if args.wandb_project else None
    train, val = Corpus.split(args.corpus, args.val_shards)
    if train.charset != init["charset"]:
        raise SystemExit(f"{args.corpus} does not have the vocabulary {args.init_from} was trained on")
    if train.tokenizer != "latex":
        raise SystemExit(f"{args.corpus} does not use the latex tokenizer, so the recognizer cannot read its labels")

    rec_ckpt = torch.load(args.recognizer, map_location=device, weights_only=False)
    recognizer = Recognizer(**rec_ckpt["config"]).to(device)
    recognizer.load_state_dict(rec_ckpt["model"])
    recognizer.eval().requires_grad_(False)
    rec_token = char_tokens(rec_ckpt["charset"])
    train_labels = recognizer_labels(train, rec_token)
    val_labels = recognizer_labels(val, rec_token)
    readable = sum(label is not None for label in train_labels)
    print(f"train {len(train)} lines ({readable} the recognizer can read), val {len(val)} lines", flush=True)

    config = ckpt["config"] if ckpt is not None else init["config"]
    if run:
        run.config.update({"model_config": config, "recognizer_step": rec_ckpt["step"]}, allow_val_change=True)
    model = FlowStudent(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step, epoch, pos, best_val = 0, 0, 0, math.inf
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, epoch, pos, best_val = ckpt["step"], ckpt["epoch"], ckpt["pos"], ckpt["best_val"]
        print(f"resumed at step {step}", flush=True)
    else:
        model.load_state_dict(init["model"])
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
        batches = train.batches(args.max_points, random.Random(args.seed * 1000 + epoch))
        while pos < len(batches) and step < args.steps:
            idx = batches[pos]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
            batch = to_device(train.batch(idx), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model.loss(batch)
                legibility = legibility_loss(model, recognizer, batch, [train_labels[i] for i in idx], rec_token,
                                             args.legibility_t_min)
            loss = total_loss(losses, args)
            if legibility is not None:
                loss = loss + args.legibility_weight * legibility
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
                if legibility is not None:
                    rec["legibility"] = legibility.item()
                diverged = False
                if step % args.eval_every == 0:
                    v = evaluate(model, recognizer, val, val_labels, rec_token, args, device)
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
    ap.add_argument("--init-from", required=True, help="flow student checkpoint to fine-tune")
    ap.add_argument("--recognizer", required=True, help="recognizer checkpoint, kept frozen")
    ap.add_argument("--corpus", help="default: the corpus the student was trained on")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--max-points", type=int, default=65536,
                    help="padded ink points per batch; each step runs the student twice")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--legibility-weight", type=float, default=0.1)
    ap.add_argument("--legibility-t-min", type=float, default=0.8,
                    help="the legibility term only sees flow times from here to 1, where the ink is nearly finished")
    ap.add_argument("--smooth-weight", type=float, default=1.0)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--char-weight", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--sample-lines", type=int, default=8, help="held-out lines drawn per sample image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-entity", default=None, help="W&B user or team (default: your W&B default entity)")
    ap.add_argument("--wandb-project", default=None, help="W&B project; W&B is not used without it")
    main(ap.parse_args())
