r"""Train the ink-to-LaTeX recognizer (scripts/recognizer_model.py) on a
MathWriting corpus (scripts/mathwriting_corpus.py, or font_corpus.py's, whose
rendered inks it reads like any other: `\typeset{label}` is trained on
`label`).

On a font corpus built over the mixed corpus (font_corpus.py --base
data/mixed_corpus), it also reads the teacher's plain text. Those lines keep
their `\text{...}` wrapper as the label, since that is what a student is asked
to write, and it keeps prose apart from math that looks the same.

Loss = token cross-entropy + --ctc-weight x CTC. Inks get a random affine
transform and point jitter, so it reads the shape rather than one writer's
proportions.

Every --eval-every steps each validation source is evaluated on its own:
human math (`val_*`), teacher text (`val_text_*`) and font renders of
MathWriting's valid labels (`val_font_*`, from font_corpus.py's
font_valid.npz). Each gets the losses, then --read-lines inks read greedily
for the token error rate (edit distance / label tokens) and the fraction read
exactly, and its --worst worst reads are drawn into <run-dir>/reads/ and W&B.
The rate on human math is how far it can be trusted as a judge of generated
ink.

The run directory, resuming and W&B work as in scripts/train_flow.py.

Usage: python scripts/train_recognizer.py --run-dir models/recognizer/<name>
           [--corpus data/font_corpus | data/font_mixed_corpus] [--steps 100000] [--max-points 65536]
           [--wandb-project P [--wandb-entity E] [--artifact none]]
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from corpus_artifact import add_wandb_args, pull, shard_names
from flow_model import ink_statistics
from ink_corpus import Corpus, split_text
from recognizer_model import Recognizer, ink_positions
from train_student import lr_at, start_wandb_run, to_device

LOSSES = ("ce", "ctc", "token_acc")
TEXT_PREFIX = "\\text{"  # mixed_corpus.py's wrapper for teacher lines
TYPESET_PREFIX = "\\typeset{"  # font_corpus.WRAPPER; importing it would need the fonts
FONT_VALID = "font_valid.npz"  # font_corpus.VALID_FILE
SOURCES = {"val_": "human math", "val_text_": "teacher text", "val_font_": "font renders"}


def unwrap(text):
    prefix = TYPESET_PREFIX
    return text[len(prefix):-1] if text.startswith(prefix) and text.endswith("}") else text


def augment(ink, ink_len):
    """A random rotation, shear and horizontal stretch per ink, and jitter of
    about a tenth of the point spacing."""
    B, N, _ = ink.shape
    device = ink.device
    angle = (torch.rand(B, device=device) * 2 - 1) * 0.1
    shear = (torch.rand(B, device=device) * 2 - 1) * 0.2
    stretch = torch.exp((torch.rand(B, device=device) * 2 - 1) * 0.15)
    cos, sin = angle.cos(), angle.sin()
    transform = torch.stack([torch.stack([cos * stretch, cos * shear - sin], -1),
                             torch.stack([sin * stretch, sin * shear + cos], -1)], -2)
    xy = ink[..., :2] @ transform.transpose(1, 2) + 0.003 * torch.randn(B, N, 2, device=device)
    mask = (torch.arange(N, device=device) < ink_len.unsqueeze(1)).unsqueeze(-1)
    return torch.cat([xy, ink[..., 2:]], dim=-1) * mask


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def inputs(batch, mu, std):
    return ink_positions(batch["offsets"], batch["ink_len"], mu, std), batch["ink_len"]


def validation_groups(texts):
    """Validation line indices by metric prefix, from the labels before
    unwrap(): human math under `val_` (its name before the other sources were
    added), teacher text under `val_text_`, font renders under `val_font_`.
    Empty groups are left out."""
    groups = {prefix: [] for prefix in SOURCES}
    for i, text in enumerate(texts):
        if text.startswith(TYPESET_PREFIX):
            groups["val_font_"].append(i)
        elif text.startswith(TEXT_PREFIX):
            groups["val_text_"].append(i)
        else:
            groups["val_"].append(i)
    return {k: v for k, v in groups.items() if v}


def draw_worst(rows, path):
    # Imported here: inspect_recognizer imports this module, and draw needs Pillow.
    from inspect_recognizer import draw

    image = draw(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return image


@torch.no_grad()
def evaluate(model, corpus, indices, args, mu, std, device, limit=200):
    """Losses over up to `limit` batches of the indexed lines, the greedy read
    rates over --read-lines of them, and the --worst worst-read inks (most
    token errors first) as inspect_recognizer.draw rows."""
    model.eval()
    totals = dict.fromkeys(LOSSES, 0.0)
    rng = random.Random(0)
    order = list(indices)
    rng.shuffle(order)
    batches = corpus.pack_batches(order, args.max_points, pool=64)
    rng.shuffle(batches)
    batches = batches[:limit]
    for idx in batches:
        batch = to_device(corpus.batch(idx), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(*inputs(batch, mu, std), batch["text"], batch["text_len"])
        for k in LOSSES:
            totals[k] += losses[k].item()
    out = {k: v / len(batches) for k, v in totals.items()}

    indices = np.asarray(indices)
    picks = indices[np.linspace(0, len(indices) - 1, min(args.read_lines, len(indices))).astype(int)]
    errors = tokens = exact = 0
    misreads = []
    for b in range(0, len(picks), 64):
        batch = to_device(corpus.batch(picks[b:b + 64]), device)
        ink, ink_len = inputs(batch, mu, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            read = model.read(ink, ink_len)
        for k, (i, ids) in enumerate(zip(picks[b:b + 64], read)):
            label = split_text(corpus.texts[i], "latex")
            target = [corpus.token[t] for t in label]
            d = edit_distance(ids, target)
            errors, tokens, exact = errors + d, tokens + len(target), exact + (d == 0)
            if d:
                misreads.append((d, d / len(target), ink[k].float().cpu().numpy(), int(ink_len[k]),
                                 " ".join(label), " ".join(corpus.charset[t - 2] for t in ids)))
    model.train()
    out.update(token_error_rate=errors / tokens, exact=exact / len(picks))
    misreads.sort(key=lambda m: (m[0], m[1]), reverse=True)
    return out, [m[2:] for m in misreads[:args.worst]]


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
            args.artifact = artifact.name
            run.config.update({"artifact": args.artifact}, allow_val_change=True)
            names = shard_names(artifact)
    train, val = Corpus.split(args.corpus, args.val_shards, names)
    if train.tokenizer != "latex":
        raise SystemExit(f"{args.corpus} does not use the latex tokenizer")
    font_valid = os.path.join(args.corpus, FONT_VALID)
    if os.path.exists(font_valid):
        val = Corpus(args.corpus, val.shards + [font_valid])
    elif any(t.startswith(TYPESET_PREFIX) for t in train.texts):
        print(f"no {font_valid}: font renders are trained on but not validated "
              "(font_corpus.py writes it)", flush=True)
    val_groups = validation_groups(val.texts)
    for corpus in (train, val):
        corpus.texts = [unwrap(t) for t in corpus.texts]
    print(f"train {len(train)} lines, val {len(val)} lines "
          f"({', '.join(f'{k}: {len(v)}' for k, v in val_groups.items())})", flush=True)
    mu = torch.tensor(train.meta["mu"], dtype=torch.float32, device=device)
    std = torch.tensor(train.meta["std"], dtype=torch.float32, device=device)

    if ckpt is not None:
        config = ckpt["config"]
    else:
        config = dict(vocab_size=len(train.charset) + 2, coord_std=ink_statistics(train)[0], d_model=args.d_model,
                      heads=args.heads, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers)
    if run:
        run.config.update({"model_config": config}, allow_val_change=True)
    model = Recognizer(**config).to(device)
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
        torch.save({"architecture": "recognizer", "model": model.state_dict(), "opt": opt.state_dict(),
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
            batch = to_device(train.batch(batches[pos]), device)
            ink, ink_len = inputs(batch, mu, std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model.loss(augment(ink, ink_len), ink_len, batch["text"], batch["text_len"])
            loss = losses["ce"] + args.ctc_weight * losses["ctc"]
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
                images = {}
                if step % args.eval_every == 0:
                    for prefix, indices in val_groups.items():
                        v, worst = evaluate(model, val, indices, args, mu, std, device)
                        rec.update({f"{prefix}{k}": x for k, x in v.items()})
                        for _, _, label, read in worst[:3]:
                            print(f"  {SOURCES[prefix]}\n  label {label!r}\n  read  {read!r}", flush=True)
                        if worst:
                            name = f"{prefix}worst"
                            path = os.path.join(args.run_dir, "reads", f"step_{step:07d}_{name}.png")
                            images[name] = (draw_worst(worst, path),
                                            f"{SOURCES[prefix]}, step {step}: {len(worst)} worst reads, "
                                            "label green, read below")
                    best_val = min(best_val, rec.get("val_ce", math.inf))
                    save()
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if run:
                    import wandb

                    run.log({**rec, **{k: wandb.Image(image, caption=caption)
                                       for k, (image, caption) in images.items()}}, step=step)
        if pos >= len(batches):
            epoch, pos = epoch + 1, 0
    save()
    print(f"done at step {step}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--corpus", default="data/font_corpus")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--max-points", type=int, default=65536, help="padded ink points per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ctc-weight", type=float, default=0.2)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--encoder-layers", type=int, default=6)
    ap.add_argument("--decoder-layers", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--read-lines", type=int, default=512,
                    help="held-out inks per source read greedily at each evaluation")
    ap.add_argument("--worst", type=int, default=10, help="worst-read inks drawn per source at each evaluation")
    ap.add_argument("--seed", type=int, default=0)
    add_wandb_args(ap, artifact_default="none",
                   artifact_help="corpus artifact name, optionally :version; none uses --corpus as it is")
    main(ap.parse_args())
