"""Show where a recognizer (scripts/train_recognizer.py) misreads held-out ink.

Reads --pool validation inks spread over the shard, prints the token error
rate, the exact-read rate and the most common token mistakes (substitutions,
tokens it drops, tokens it adds), and draws --n misread inks, one per row,
with the label and the reading under each. --all draws inks whether or not
they were misread.

With --wandb-project, the image, the two rates and tables of every mistake
and every misread ink go to a new W&B run (job type "inspect").

Usage: python scripts/inspect_recognizer.py --checkpoint models/recognizer/<run>/checkpoint.pt
           [--corpus DIR] [--pool 2000] [--n 24] [--all] [--out PATH]
           [--wandb-project P [--wandb-entity E]]
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import torch

from ink_corpus import Corpus, split_text
from recognizer_model import Recognizer
from train_recognizer import inputs, to_device, unwrap

INK_PX = 110      # height of each ink in the image
MAX_WIDTH = 1600  # px; wider ink is cut off


def edit_ops(read, target):
    """The token mistakes of a minimal alignment: ("sub", target, read),
    ("drop", target) and ("add", read)."""
    n, m = len(read), len(target)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (read[i - 1] != target[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (read[i - 1] != target[j - 1]):
            if read[i - 1] != target[j - 1]:
                ops.append(("sub", target[j - 1], read[i - 1]))
            i, j = i - 1, j - 1
        elif j and d[i, j] == d[i, j - 1] + 1:
            ops.append(("drop", target[j - 1]))
            j -= 1
        else:
            ops.append(("add", read[i - 1]))
            i -= 1
    return ops


def draw(rows):
    """rows: (ink (n, 3) in ink heights, its length, label, reading). One
    image, an ink and its two lines of text per row."""
    from PIL import Image, ImageDraw

    pad, text_px = 10, 14
    tiles = []
    for ink, length, label, read in rows:
        xy, pen = ink[:length, :2], ink[:length, 2]
        xy = (xy - xy.min(axis=0)) * INK_PX  # y points down, as in MathWriting
        width = int(min(MAX_WIDTH, xy[:, 0].max() + 2 * pad))
        tile = Image.new("RGB", (max(width, 8 * max(len(label), len(read)) + 2 * pad, 200),
                                 INK_PX + 2 * pad + 2 * text_px + pad), "white")
        canvas = ImageDraw.Draw(tile)
        start = 0
        for k in range(length):
            if pen[k] > 0 or k == length - 1:
                stroke = [(pad + x, pad + y) for x, y in xy[start:k + 1]]
                if len(stroke) > 1:
                    canvas.line(stroke, fill="black", width=2)
                else:
                    canvas.point(stroke[0], fill="black")
                start = k + 1
        y = INK_PX + 2 * pad
        canvas.text((pad, y), f"label: {label}", fill=(0, 110, 0))
        canvas.text((pad, y + text_px), f"read:  {read}", fill=(0, 0, 0) if read == label else (190, 0, 0))
        canvas.line([(0, tile.height - 1), (tile.width, tile.height - 1)], fill=(200, 200, 200))
        tiles.append(tile)
    image = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)), "white")
    y = 0
    for tile in tiles:
        image.paste(tile, (0, y))
        y += tile.height
    return image


@torch.no_grad()
def main(args):
    device = "cuda"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Recognizer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    train_args = ckpt["args"]
    corpus_dir = args.corpus or train_args["corpus"]
    shards = sorted(glob.glob(os.path.join(corpus_dir, "shard_*.npz")))
    val = Corpus(corpus_dir, shards[-train_args.get("val_shards", 1):])
    if val.charset[:len(ckpt["charset"])] != ckpt["charset"]:
        raise SystemExit(f"{corpus_dir} is not in the vocabulary {args.checkpoint} was trained on")
    val.texts = [unwrap(t) for t in val.texts]
    mu = torch.tensor(val.meta["mu"], dtype=torch.float32, device=device)
    std = torch.tensor(val.meta["std"], dtype=torch.float32, device=device)

    picks = np.linspace(0, len(val) - 1, min(args.pool, len(val))).astype(int)
    mistakes, rows, misreads = Counter(), [], []
    errors = tokens = exact = 0
    for b in range(0, len(picks), 64):
        batch = to_device(val.batch(picks[b:b + 64]), device)
        ink, ink_len = inputs(batch, mu, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reads = model.read(ink, ink_len)
        for k, (i, ids) in enumerate(zip(picks[b:b + 64], reads)):
            target = split_text(val.texts[i], "latex")
            read = [val.charset[t - 2] for t in ids]
            ops = edit_ops(read, target)
            mistakes.update(ops)
            if ops:
                misreads.append((" ".join(target), " ".join(read), len(ops)))
            errors, tokens, exact = errors + len(ops), tokens + len(target), exact + (not ops)
            if (ops or args.all) and len(rows) < args.n:
                rows.append((ink[k].float().cpu().numpy(), int(ink_len[k]), " ".join(target), " ".join(read)))

    rates = {"token_error_rate": errors / tokens, "exact": exact / len(picks)}
    print(f"{len(picks)} inks from {corpus_dir}: token error rate {rates['token_error_rate']:.3f}, "
          f"read exactly {rates['exact']:.1%}")
    print("most common mistakes (sub: label -> read; drop: in the label, not read; add: read, not in the label):")
    for op, count in mistakes.most_common(args.top):
        print(f"  {count:5d}  {op[0]:4s} " + " -> ".join(op[1:]))
    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    image = None
    if rows:
        out = args.out or os.path.join("artifacts", "recognizer", run_name + "_reads.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        image = draw(rows)
        image.save(out)
        print(f"drew {len(rows)} inks: {out}")

    if args.wandb_project:
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="inspect",
                         name=f"{run_name}-reads",
                         config={**vars(args), "checkpoint_step": ckpt["step"], "corpus": corpus_dir})
        mistake_rows = [[op[0], op[1], op[2] if op[0] == "sub" else "", count] if op[0] != "add"
                        else [op[0], "", op[1], count] for op, count in mistakes.most_common()]
        record = {**rates,
                  "mistakes": wandb.Table(columns=["kind", "label token", "read token", "count"], data=mistake_rows),
                  "misreads": wandb.Table(columns=["label", "read", "token errors"], data=misreads)}
        if image is not None:
            record["reads"] = wandb.Image(image, caption=f"{run_name} step {ckpt['step']}: label green, read below")
        run.log(record)
        run.summary.update(rates)
        run.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", help="corpus directory (default: the one the checkpoint was trained on)")
    ap.add_argument("--pool", type=int, default=2000, help="validation inks read")
    ap.add_argument("--n", type=int, default=24, help="inks drawn")
    ap.add_argument("--all", action="store_true", help="draw inks whether or not they were misread")
    ap.add_argument("--top", type=int, default=25, help="mistakes listed")
    ap.add_argument("--out", help="image path (default: artifacts/recognizer/<run>_reads.png)")
    ap.add_argument("--wandb-entity", default=None, help="W&B user or team (default: your W&B default entity)")
    ap.add_argument("--wandb-project", default=None, help="W&B project; W&B is not used without it")
    main(ap.parse_args())
