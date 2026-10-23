"""Test whether a flow student's misshapen ink (loops, scribbles, repeated
symbols) comes from being given the wrong number of points.

FlowStudent.sample() draws the point count from the length head before any
denoising, but training always uses the true count. This writes the same
held-out labels from the same noise at several point counts: the true one,
the true one scaled by each --scales factor, the length head's most likely
bin and a draw from it (what sample() does). The recognizer reads every ink
and the token error rate is reported per count, with its mistakes split into
added, dropped and substituted tokens (repeated symbols show up as added).

If the error rate is lowest at the true count and climbs as the count moves
away from it, and the sampled count is often far from the true one, the
length is the problem. If the true count is about as bad as the sampled one,
it is not, and the points are being misplaced within a correct budget.

Everything goes to a new W&B run (job type "length_sweep"): a table of the
rates per count, the error rate against the scale factor, a histogram of
sampled / true counts, and for --draw labels an image with the real ink and
every count's ink under it, each with what the recognizer read.

Usage: python scripts/length_sweep.py --student models/student/<run>/checkpoint.pt
           --recognizer models/recognizer/<run>/checkpoint.pt --wandb-project P [--wandb-entity E]
           [--corpus DIR] [--lines 500] [--scales 0.6 0.8 0.9 1.1 1.25 1.5] [--draw 8]
           [--steps S] [--guidance G] [--bias 0] [--batch 16] [--seed 0]
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from flow_model import BIAS_TEMPERATURE, LENGTH_BIN, LENGTH_BINS
from ink_corpus import Corpus, char_tokens, encode_texts, split_text
from inspect_recognizer import draw
from measure_legibility import read_inks, score
from recognizer_model import Recognizer, ink_positions
from sample_student import load_model
from train_recognizer import inputs, to_device, unwrap


def condition_lengths(model, text, text_len, true_len, scales, bias, generator):
    """{condition name: point counts (B,)} for one batch of labels."""
    _, _, length_logit = model.encode_text(text, text_len)
    sampled_bin = torch.multinomial(F.softmax(length_logit * (1 + bias), dim=-1), 1, generator=generator).squeeze(1)
    lengths = {"true": true_len,
               "argmax": length_logit.argmax(-1) * LENGTH_BIN + LENGTH_BIN // 2,
               "sampled": sampled_bin * LENGTH_BIN + LENGTH_BIN // 2}
    for s in scales:
        lengths[f"true x{s:g}"] = (true_len.float() * s).round().long().clamp(LENGTH_BIN, LENGTH_BIN * LENGTH_BINS)
    return lengths


def write(model, noise, text, text_len, ink_len, mu, std):
    """The student's ink at the given point counts, in the recognizer's input
    form. Every count starts from the same noise, cut to its length."""
    x, _ = model.transport(noise[:, :int(ink_len.max())].clone(), text, text_len, ink_len)
    offsets = model.to_offsets(x)
    # As in FlowStudent.sample(): the last point always ends a stroke.
    offsets[torch.arange(len(ink_len)), ink_len - 1, 2] = 1.0
    return ink_positions(offsets, ink_len, mu, std)


def rates(scored, targets):
    tokens = sum(len(t) for t in targets)
    kinds = Counter(op[0] for _, ops in scored for op in ops)
    return {"token_error_rate": sum(len(ops) for _, ops in scored) / tokens,
            "exact": sum(not ops for _, ops in scored) / len(scored),
            "added_per_token": kinds["add"] / tokens,
            "dropped_per_token": kinds["drop"] / tokens,
            "substituted_per_token": kinds["sub"] / tokens}


@torch.no_grad()
def main(args):
    device = "cuda"
    student, student_ckpt = load_model(args.student, device)
    if student_ckpt.get("architecture") != "flow":
        raise SystemExit(f"{args.student} is not a flow student")
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
    tokenizer = student_ckpt["meta"].get("tokenizer", "char")
    student_token = char_tokens(student_ckpt["charset"])
    readable = set(rec_ckpt["charset"])
    candidates = [i for i, t in enumerate(val.texts)
                  if set(split_text(t, tokenizer)) <= set(student_token) and set(split_text(t, tokenizer)) <= readable]
    picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(args.lines, len(candidates))).astype(int)]
    targets = [split_text(val.texts[i], tokenizer) for i in picks]
    print(f"{len(picks)} labels from {corpus_dir} ({len(candidates)} of {len(val)} in both vocabularies)", flush=True)

    name = os.path.basename(os.path.dirname(os.path.abspath(args.student)))
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="length_sweep",
                     name=f"{name}-length-sweep",
                     config={**vars(args), "student_step": student_ckpt["step"], "recognizer_step": rec_ckpt["step"],
                             "corpus": corpus_dir, "sample_steps": student.sample_steps,
                             "sample_guidance": student.sample_guidance})

    mu = torch.tensor(val.meta["mu"], dtype=torch.float32, device=device)
    std = torch.tensor(val.meta["std"], dtype=torch.float32, device=device)
    student_mu = torch.tensor(student_ckpt["meta"]["mu"], dtype=torch.float32, device=device)
    student_std = torch.tensor(student_ckpt["meta"]["std"], dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    scales = [s for s in args.scales if s != 1.0]
    drawn = set(np.linspace(0, len(picks) - 1, min(args.draw, len(picks))).astype(int).tolist())

    scored = {}       # condition -> per label (read tokens, edit ops)
    lengths = {}      # condition -> per label point count
    real_scored = []
    tiles = []
    for b in range(0, len(picks), args.batch):
        idx = picks[b:b + args.batch]
        batch = to_device(val.batch(idx), device)
        text, text_len, true_len = batch["text"], batch["text_len"], batch["ink_len"]
        real_ink, _ = inputs(batch, mu, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            real_reads = recognizer.read(real_ink, true_len)
        real = score(real_reads, targets[b:b + len(idx)], rec_ckpt["charset"])
        real_scored += real

        counts = condition_lengths(student, text, text_len, true_len, scales, args.bias, gen)
        longest = max(int(n.max()) for n in counts.values())
        noise = torch.randn((len(idx), longest, 3), generator=gen, device=device)
        noise = noise * torch.exp(torch.tensor(-BIAS_TEMPERATURE * args.bias, device=device))
        inks = {}
        for cond, ink_len in counts.items():
            ink = write(student, noise, text, text_len, ink_len, student_mu, student_std)
            reads = read_inks(recognizer, ink, ink_len)
            scored.setdefault(cond, []).extend(score(reads, targets[b:b + len(idx)], rec_ckpt["charset"]))
            lengths.setdefault(cond, []).extend(ink_len.tolist())
            inks[cond] = (ink.float().cpu().numpy(), ink_len.tolist())

        for k in range(len(idx)):
            if b + k not in drawn:
                continue
            label = " ".join(targets[b + k])
            rows = [(real_ink[k].float().cpu().numpy(), int(true_len[k]), f"real ink, {int(true_len[k])} points",
                     " ".join(real[k][0]))]
            for cond, (ink, ink_len) in inks.items():
                rows.append((ink[k], ink_len[k], f"{cond}, {ink_len[k]} points", " ".join(scored[cond][b + k][0])))
            tiles.append(wandb.Image(draw(rows), caption=f"{label}: real ink, then the student at each point count"))
        print(f"{min(b + args.batch, len(picks))}/{len(picks)} labels", flush=True)

    true_n = np.array(lengths["true"], dtype=float)
    ratio = np.array(lengths["sampled"]) / true_n
    table = wandb.Table(columns=["condition", "scale", "token_error_rate", "exact", "added_per_token",
                                 "dropped_per_token", "substituted_per_token", "median count / true"])
    summary = {}
    real_rates = rates(real_scored, targets)
    table.add_data("real ink", None, *real_rates.values(), 1.0)
    print(f"{'condition':>12s}  error   exact   added  dropped  subst  count/true")
    print(f"{'real ink':>12s}  {real_rates['token_error_rate']:.3f}  {real_rates['exact']:6.1%}")
    order = ["argmax", "sampled"] + sorted((c for c in scored if c.startswith("true")),
                                           key=lambda c: float(c.split("x")[-1]) if "x" in c else 1.0)
    for cond in order:
        r = rates(scored[cond], targets)
        scale = float(cond.split("x")[-1]) if cond.startswith("true x") else (1.0 if cond == "true" else None)
        median = float(np.median(np.array(lengths[cond]) / true_n))
        table.add_data(cond, scale, *r.values(), median)
        summary.update({f"{cond}/{k}": v for k, v in r.items()})
        print(f"{cond:>12s}  {r['token_error_rate']:.3f}  {r['exact']:6.1%}  {r['added_per_token']:.3f}  "
              f"{r['dropped_per_token']:.3f}    {r['substituted_per_token']:.3f}  {median:.2f}")

    # How far the length head's draws land from the true count.
    summary.update({"sampled_within_10pct": float(np.mean(np.abs(ratio - 1) <= 0.1)),
                    "sampled_within_25pct": float(np.mean(np.abs(ratio - 1) <= 0.25)),
                    "sampled_mean_abs_log_ratio": float(np.mean(np.abs(np.log(ratio))))})
    print(f"sampled count within 10% of the true one: {summary['sampled_within_10pct']:.1%}, "
          f"within 25%: {summary['sampled_within_25pct']:.1%}")

    # The sampled condition's error by how far its count missed, label by label.
    sampled_errors = np.array([len(ops) / len(t) for (_, ops), t in zip(scored["sampled"], targets)])
    miss = wandb.Table(columns=["sampled / true", "token_error_rate"],
                       data=[[float(r), float(e)] for r, e in zip(ratio, sampled_errors)])
    by_scale = wandb.Table(columns=["scale", "token_error_rate"],
                           data=[[s, summary[f"{c}/token_error_rate"]] for c, s in
                                 sorted([("true", 1.0)] + [(f"true x{s:g}", s) for s in scales],
                                        key=lambda cs: cs[1])])
    run.log({"rates": table,
             "error_vs_scale": wandb.plot.line(by_scale, "scale", "token_error_rate",
                                               title="Token error rate vs point count / true count"),
             "sampled_error_vs_miss": wandb.plot.scatter(miss, "sampled / true", "token_error_rate",
                                                         title="Sampled count: error vs how far the count missed"),
             "sampled_over_true": wandb.Histogram(ratio),
             "samples": tiles})
    run.summary.update({**summary, **{f"real/{k}": v for k, v in real_rates.items()}})
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True, help="flow student checkpoint")
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--corpus", help="corpus whose validation labels are written (default: the student's)")
    ap.add_argument("--lines", type=int, default=500, help="labels written, spread over the validation shard")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 0.9, 1.1, 1.25, 1.5],
                    help="factors on the true point count")
    ap.add_argument("--draw", type=int, default=8, help="labels drawn at every point count")
    ap.add_argument("--steps", type=int, help="Euler steps (default: the checkpoint's)")
    ap.add_argument("--guidance", type=float, help="guidance scale (default: the checkpoint's)")
    ap.add_argument("--bias", type=float, default=0.0, help="sampling bias; MathWriting corpora are written at 0")
    ap.add_argument("--batch", type=int, default=16,
                    help="labels written at once; attention over ~2000 points grows with the square")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-entity", default=None, help="W&B user or team (default: your W&B default entity)")
    ap.add_argument("--wandb-project", required=True)
    main(ap.parse_args())
