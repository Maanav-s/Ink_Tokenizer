"""Train the style-conditioned spline flow student (scripts/style_flow_model.py)
on a spline corpus whose shards have `writers` (scripts/pool_corpus.py, then
scripts/spline_corpus.py).

As scripts/train_spline_flow.py, plus each batch's reference sets: for a line
whose writer has other lines in the same split, up to --refs of them (each
a random window of at most --ref-points control points); otherwise a crop of
--crop of the line itself. Validation lines get references from validation
lines only, and DeepWriting's and BRUSH's validation writers are unseen in
training, so val_flow_null - val_flow is what a new writer's style is worth.

The trained loss adds beta times the KL and --contrastive-weight times the
supervised contrastive loss. beta is not a setting: it adapts every step so
the KL, the nats z carries, stays near --kl-target (log beta moves by
--beta-lr times the KL's relative excess over the target), as in a
constrained VAE. Too low a target leaves no room for style (val_style_gain
stays near 0); too high lets z copy the references' content.

Each evaluation also reports:
- val_id_acc: of held-out lines embedded on their own, how often the nearest
  other line of the same source (cosine of the projection head's output) is
  by the same writer,
  and val_id_chance, what picking at random would score;
- grad_flow, grad_kl, grad_contrastive: the norm of each weighted term's
  gradient on the style encoder, for one training batch, to see which term
  drives it.

Sample images show, per held-out writer, the real line, the student without
a style, and the student with the style of that writer's other lines.

--init-from starts from a plain spline flow checkpoint (train_spline_flow.py)
on the same normalization: its weights and architecture, with the style
encoder new and the style's effect starting at zero.

DeepWriting's license forbids distribution: with it in the corpus, sample
images in W&B show its ink.

Usage: python scripts/train_style_flow.py --run-dir models/student/<name>
           --corpus data/pooled_spline_corpus [--init-from models/student/<plain>/checkpoint.pt]
           [--steps 50000] [--accumulate 2] [--wandb-project P [--wandb-entity E]]
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from corpus_artifact import add_wandb_args
from flow_model import ink_statistics
from ink_corpus import Corpus, encode_texts
from sample_student import preview_scale
from style_flow_model import StyleSplineFlowStudent
from text_to_ink import ink_extent, render, to_line
from train_flow import DIVERGED, LOSSES, total_loss
from train_student import lr_at, start_wandb_run, to_device

STYLE_LOSSES = (*LOSSES, "kl", "contrastive")
LOG_BETA_RANGE = (math.log(1e-6), math.log(10.0))


def pad(inks):
    lengths = [len(o) for o in inks]
    out = torch.zeros(len(inks), max(lengths, default=1), 3)
    for b, o in enumerate(inks):
        out[b, :len(o)] = torch.from_numpy(o)
    return out, torch.tensor(lengths, dtype=torch.long)


class References:
    """Reference sets and writer labels for batches of one corpus."""

    def __init__(self, corpus, refs, crop, max_points):
        self.corpus, self.refs, self.crop, self.max_points = corpus, refs, crop, max_points
        self.lines = {}
        for i, w in enumerate(corpus.writers):
            if w:
                self.lines.setdefault(w, []).append(i)
        self.writer_id = {w: k for k, w in enumerate(sorted(self.lines))}
        self.source_id = {s: k for k, s in enumerate(sorted({w.split("/")[0] for w in self.lines}))}

    def window(self, i, n, rng):
        s, k = self.corpus.starts[i], int(self.corpus.lengths[i])
        n = min(n, k)
        a = s + int(rng.integers(0, k - n + 1))
        return self.corpus.offsets[a:a + n]

    def others(self, i, rng):
        """Up to self.refs other lines by line i's writer, or none."""
        pool = self.lines.get(self.corpus.writers[i], ())
        if len(pool) < 2:
            return []
        picks = rng.choice(len(pool), min(self.refs + 1, len(pool)), replace=False)
        return [pool[p] for p in picks if pool[p] != i][:self.refs]

    def __call__(self, indices, rng):
        refs, owner, writer, source, anchors = [], [], [], [], []
        for b, i in enumerate(indices):
            others = self.others(i, rng)
            if others:
                refs += [self.window(j, self.max_points, rng) for j in others]
                owner += [b] * len(others)
                anchors.append(self.window(i, self.max_points, rng))
                w = self.corpus.writers[i]
                writer.append(self.writer_id[w])
                source.append(self.source_id[w.split("/")[0]])
            else:
                n = max(2, round(self.crop * int(self.corpus.lengths[i])))
                refs.append(self.window(i, min(n, self.max_points), rng))
                owner.append(b)
                writer.append(-1)
                source.append(-1)
        ref_offsets, ref_len = pad(refs)
        anchor_offsets, anchor_len = pad(anchors)
        return {"ref_offsets": ref_offsets, "ref_len": ref_len, "ref_owner": torch.tensor(owner),
                "anchor_offsets": anchor_offsets, "anchor_len": anchor_len,
                "writer": torch.tensor(writer), "source": torch.tensor(source)}


def style_total_loss(losses, args, beta):
    return total_loss(losses, args) + beta * losses["kl"] + args.contrastive_weight * losses["contrastive"]


def update_log_beta(log_beta, kl, args):
    log_beta += args.beta_lr * (kl - args.kl_target) / args.kl_target
    return min(max(log_beta, LOG_BETA_RANGE[0]), LOG_BETA_RANGE[1])


@torch.no_grad()
def identification(model, corpus, references, device, limit=2000, batch=256):
    """{"id_acc", "id_chance"} over up to `limit` lines whose writer has
    other lines in the corpus (see the module docstring), or {} if none."""
    lines = [i for idx in references.lines.values() if len(idx) > 1 for i in idx]
    if not lines:
        return {}
    rng = np.random.default_rng(0)
    lines = [lines[k] for k in sorted(rng.permutation(len(lines))[:limit])]
    model.eval()
    embedded = []
    for b in range(0, len(lines), batch):
        offsets, ink_len = pad([references.window(i, references.max_points, rng) for i in lines[b:b + batch]])
        embedded.append(model.identity(offsets.to(device), ink_len.to(device)))
    model.train()
    e = F.normalize(torch.cat(embedded), dim=-1)
    writers = [corpus.writers[i] for i in lines]
    writer = torch.tensor([references.writer_id[w] for w in writers], device=device)
    source = torch.tensor([references.source_id[w.split("/")[0]] for w in writers], device=device)
    eye = torch.eye(len(e), dtype=torch.bool, device=device)
    candidate = (source.unsqueeze(0) == source.unsqueeze(1)) & ~eye
    same = (writer.unsqueeze(0) == writer.unsqueeze(1)) & ~eye
    scored = same.any(dim=1)
    if not scored.any():
        return {}
    nearest = (e @ e.T).masked_fill(~candidate, -torch.inf).argmax(dim=1)
    accuracy = (writer[nearest] == writer)[scored].float().mean().item()
    chance = (same.sum(1) / candidate.sum(1).clamp(min=1))[scored].mean().item()
    return {"id_acc": accuracy, "id_chance": chance}


def encoder_grad_norms(model, batch, args, beta):
    """Norm of each weighted loss term's gradient on the style encoder."""
    params = list(model.style_encoder.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model.loss(batch)
    terms = {"flow": total_loss(losses, args), "kl": beta * losses["kl"],
             "contrastive": args.contrastive_weight * losses["contrastive"]}
    out = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        out[f"grad_{name}"] = math.sqrt(sum(g.float().pow(2).sum().item() for g in grads if g is not None))
    return out


@torch.no_grad()
def evaluate(model, corpus, references, max_points, device, limit=200):
    model.eval()
    keys = (*STYLE_LOSSES, "flow_null")
    totals = dict.fromkeys(keys, 0.0)
    batches = corpus.batches(max_points, random.Random(0))[:limit]
    rng = np.random.default_rng(0)
    # The loss draws noise and flow times; fixing them makes evaluations comparable.
    generator = torch.Generator(device=device).manual_seed(0)
    for idx in batches:
        batch = {**corpus.batch(idx), **references(idx, rng)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(to_device(batch, device), generator)
        for k in keys:
            totals[k] += losses[k].item()
    model.train()
    out = {k: v / len(batches) for k, v in totals.items()}
    out["style_gain"] = out["flow_null"] - out["flow"]
    return out


@torch.no_grad()
def style_samples(model, corpus, source, references, n, seed=0, guidances=(1.0,), prior=0):
    """Rows per held-out writer: the source corpus's ink for one of their
    lines, the student without a style, and with the style of their other
    lines at each style guidance in `guidances` (all with the same noise).
    Then the first writer's line `prior` times, each with a z drawn from the
    prior N(0, I): new styles. None if no line has a known writer."""
    first = {}
    for i, w in enumerate(corpus.writers):
        if w and len(references.lines[w]) > 1:
            first.setdefault(w, i)
    picks = [first[w] for w in sorted(first)]
    picks = picks[::max(1, len(picks) // n)][:n]
    if not picks:
        return None
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    refs = to_device(references(picks, np.random.default_rng(seed)), device)
    mu, _ = model.encode_style(refs["ref_offsets"], refs["ref_len"], refs["ref_owner"], len(picks))
    texts = [corpus.texts[i] for i in picks]
    text, text_len = encode_texts(texts, corpus.token, corpus.tokenizer)
    text, text_len = text.to(device), text_len.to(device)
    plain = model.sample(text, text_len, 0.0, generator=torch.Generator(device=device).manual_seed(seed))
    styled = [model.sample(text, text_len, 0.0, generator=torch.Generator(device=device).manual_seed(seed),
                           cond=model.style_cond(mu), style_guidance=g) for g in guidances]
    drawn = []
    if prior:
        z = torch.randn(prior, mu.shape[1], generator=torch.Generator(device=device).manual_seed(seed),
                        device=device)
        drawn = model.sample(text[:1].expand(prior, -1), text_len[:1].expand(prior),
                             0.0, generator=torch.Generator(device=device).manual_seed(seed),
                             cond=model.style_cond(z))
    model.train(was_training)

    mu_, std_ = np.array(source.meta["mu"]), np.array(source.meta["std"])
    scale = preview_scale(source.meta)
    rows = []
    for b, (i, text_i) in enumerate(zip(picks, texts)):
        a, k = source.starts[i], source.lengths[i]
        rows.append(to_line(text_i, source.offsets[a:a + k], source.chars[a:a + k], mu_, std_, scale))
        rows += [to_line(text_i, off.numpy(), chars.numpy(), mu_, std_, scale)
                 for off, chars, _ in (plain[b], *(s[b] for s in styled))]
    real = [ink_extent(r) for r in rows[::2 + len(guidances)]]
    rows += [to_line(texts[0], off.numpy(), chars.numpy(), mu_, std_, scale) for off, chars, _ in drawn]
    return render(rows, max_width=1.5 * max(e[0] for e in real),
                  max_height=1.5 * max(max(-e[1], e[2]) for e in real))


def build_config(args, train):
    style = dict(z_dim=args.z_dim, style_d_model=args.style_d_model, style_layers=args.style_layers,
                 style_dropout=args.style_dropout, temperature=args.temperature,
                 projection_dim=args.projection_dim)
    if args.init_from:
        plain = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if plain.get("architecture") != "spline_flow":
            raise ValueError(f"{args.init_from} is not a spline flow checkpoint")
        for key in ("mu", "std"):
            if plain["meta"][key] != train.meta[key]:
                raise ValueError(f"{args.init_from} was trained on {key} {plain['meta'][key]}, "
                                 f"{args.corpus} has {train.meta[key]}")
        if plain["config"]["points_per_control"] != train.meta["spline"]["points_per_control"]:
            raise ValueError(f"{args.init_from} and {args.corpus} differ in points_per_control")
        if train.charset[:len(plain["charset"])] != plain["charset"]:
            raise ValueError(f"{args.corpus}'s vocabulary does not extend {args.init_from}'s")
        return {**plain["config"], "vocab_size": len(train.charset) + 2, **style}, plain["model"]
    coord_std, typical_height = ink_statistics(train)
    return dict(points_per_control=train.meta["spline"]["points_per_control"],
                vocab_size=len(train.charset) + 2, offset_mu=train.meta["mu"], offset_std=train.meta["std"],
                coord_std=coord_std, typical_height=typical_height, d_model=args.d_model,
                n_layers=args.n_layers, heads=args.heads, text_layers=args.text_layers,
                text_dropout=args.text_dropout, sample_steps=args.sample_steps,
                sample_length=args.sample_length, **style), None


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
    train_refs = References(train, args.refs, args.crop, args.ref_points)
    val_refs = References(val, args.refs, args.crop, args.ref_points)
    print(f"train {len(train)} lines, {len(train_refs.lines)} writers; "
          f"val {len(val)} lines, {len(val_refs.lines)} writers", flush=True)
    if not train_refs.lines:
        raise ValueError(f"{args.corpus} has no writers; pool it with the current pool_corpus.py")

    plain_state = None
    if ckpt is not None:
        config = ckpt["config"]
    else:
        config, plain_state = build_config(args, train)
    if run:
        run.config.update({"model_config": config}, allow_val_change=True)
    model = StyleSplineFlowStudent(**config).to(device)
    if plain_state is not None:
        print(f"initialized from {args.init_from}; new: {model.init_from_plain(plain_state)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step, epoch, pos, best_val, log_beta = 0, 0, 0, math.inf, math.log(args.beta_init)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, epoch, pos, best_val = ckpt["step"], ckpt["epoch"], ckpt["pos"], ckpt["best_val"]
        log_beta = ckpt["log_beta"]
        print(f"resumed at step {step}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters; {model.config}", flush=True)

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"architecture": "style_spline_flow", "model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "epoch": epoch, "pos": pos, "best_val": best_val, "log_beta": log_beta,
                    "config": model.config,
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
            losses = dict.fromkeys(STYLE_LOSSES, 0.0)
            for m, idx in enumerate(micro):
                rng = np.random.default_rng((args.seed, step, m))
                batch = {**train.batch(idx), **train_refs(idx, rng)}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    batch_losses = model.loss(to_device(batch, device), t_mean=args.t_mean)
                loss = style_total_loss(batch_losses, args, math.exp(log_beta))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                (loss / len(micro)).backward()
                for k in STYLE_LOSSES:
                    losses[k] += batch_losses[k].detach() / len(micro)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at step {step}")
            opt.step()
            log_beta = update_log_beta(log_beta, float(losses["kl"]), args)
            step += 1
            pos += len(micro)

            if step % args.log_every == 0 or step % args.eval_every == 0:
                rec = {"step": step, "epoch": epoch, **{k: float(losses[k]) for k in STYLE_LOSSES},
                       "grad_norm": grad_norm.item(), "lr": opt.param_groups[0]["lr"], "beta": math.exp(log_beta),
                       "elapsed_s": round(time.time() - start)}
                diverged = False
                if step % args.eval_every == 0:
                    v = evaluate(model, val, val_refs, args.max_points, device)
                    v.update(identification(model, val, val_refs, device))
                    probe = {**train.batch(micro[-1]), **train_refs(micro[-1], np.random.default_rng(step))}
                    rec.update(encoder_grad_norms(model, to_device(probe, device), args, math.exp(log_beta)))
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
                img = style_samples(model, val, source_val, val_refs, args.sample_lines, prior=4)
                if img is not None:
                    os.makedirs(os.path.join(args.run_dir, "samples"), exist_ok=True)
                    img.save(os.path.join(args.run_dir, "samples", f"step_{step:07d}.png"))
                    if run:
                        run.log({"samples": wandb.Image(img, caption="per writer: real, no style, their style; "
                                                                     "last 4 rows: random styles")},
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
    ap.add_argument("--corpus", default="data/pooled_spline_corpus")
    ap.add_argument("--init-from", help="a plain spline flow checkpoint to start from")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--max-points", type=int, default=32768,
                    help="padded control points per batch, not counting references")
    ap.add_argument("--accumulate", type=int, default=1, help="batches whose gradients are summed per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=1.0)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--char-weight", type=float, default=0.1)
    ap.add_argument("--t-mean", type=float, default=0.0,
                    help="training flow times are sigmoid(N(t_mean, 1)); positive trains nearer the finished ink. "
                         "Validation always uses 0, so runs stay comparable")
    ap.add_argument("--kl-target", type=float, default=16.0, help="nats z should carry; beta adapts to hold it")
    ap.add_argument("--beta-init", type=float, default=1e-3, help="the KL weight at step 0")
    ap.add_argument("--beta-lr", type=float, default=1e-3,
                    help="how fast log beta moves per step, per unit of relative KL error")
    ap.add_argument("--contrastive-weight", type=float, default=0.05)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--style-d-model", type=int, default=256)
    ap.add_argument("--style-layers", type=int, default=4)
    ap.add_argument("--style-dropout", type=float, default=0.1,
                    help="fraction of lines trained with no style, which is how the student writes without one")
    ap.add_argument("--temperature", type=float, default=0.1, help="of the contrastive loss")
    ap.add_argument("--projection-dim", type=int, default=64,
                    help="size of the head the contrastive loss acts through; 0 puts it on z's mean directly")
    ap.add_argument("--refs", type=int, default=4, help="other lines by the same writer per line")
    ap.add_argument("--crop", type=float, default=0.3,
                    help="fraction of a line used as its own reference when its writer is unknown")
    ap.add_argument("--ref-points", type=int, default=128, help="control points per reference at most")
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
    ap.add_argument("--sample-lines", type=int, default=8, help="held-out writers drawn per sample image")
    ap.add_argument("--seed", type=int, default=0)
    add_wandb_args(ap, artifact_default="none", artifact_help="unused: spline corpora are converted locally")
    main(ap.parse_args())
