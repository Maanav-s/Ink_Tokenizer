"""Compare style guidance scales for a style student (scripts/train_style_flow.py)
on its corpus's held-out writers: an image, and a style-match score.

Style guidance pushes the velocity away from the no-style one
(StyleSplineFlowStudent.transport), to amplify a style that z carries but the
student under-expresses.

Style match: for up to --lines-per-writer lines of each held-out writer, the
student writes the line with the style of that writer's other lines, and the
style encoder embeds the result. It scores a hit if the nearest writer
centroid of the same source (cosine; centroids from lines not being written)
is that writer. Real lines and no-style samples are scored the same way, as
upper and lower references. The encoder judges, so guidance can raise the
score with ink that only looks right to it: check the image too.

DeepWriting's license forbids distribution: the image shows its ink.

Usage: python scripts/sample_style.py --checkpoint models/student/<run>/checkpoint.pt
           [--guidance 1 2 3] [--out style_guidance.png] [--wandb-project P [--wandb-entity E]]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from flow_model import FlowStudent
from ink_corpus import Corpus, encode_texts
from sample_student import load_model
from train_style_flow import References, pad, style_samples


@torch.no_grad()
def embed(model, inks, device, batch=256):
    out = []
    for b in range(0, len(inks), batch):
        offsets, ink_len = pad(inks[b:b + batch])
        out.append(F.normalize(model.identity(offsets.to(device), ink_len.to(device)), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def style_match(model, corpus, references, guidances, lines_per_writer, seed=0):
    """{"real", "no style", g...: hit rate} over the held-out writers' lines."""
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    writers = sorted(w for w, idx in references.lines.items() if len(idx) > lines_per_writer + 1)
    targets, centroids = [], []
    for w in writers:
        idx = list(rng.permutation(references.lines[w]))
        targets += idx[:lines_per_writer]
        rest = [references.window(i, references.max_points, rng) for i in idx[lines_per_writer:][:32]]
        centroids.append(F.normalize(embed(model, rest, device).mean(dim=0), dim=0))
    centroids = torch.stack(centroids)
    writer = torch.tensor([writers.index(corpus.writers[i]) for i in targets], device=device)
    source = [w.split("/")[0] for w in writers]
    same_source = torch.tensor([[a == b for b in source] for a in source], device=device)[writer]

    def hit_rate(embedded):
        nearest = (embedded @ centroids.T).masked_fill(~same_source, -torch.inf).argmax(dim=1)
        return (nearest == writer).float().mean().item()

    real = [corpus.offsets[corpus.starts[i]:corpus.starts[i] + corpus.lengths[i]] for i in targets]
    scores = {"real": hit_rate(embed(model, real, device))}
    refs = {k: v.to(device) for k, v in references(targets, rng).items()}
    mu, _ = model.encode_style(refs["ref_offsets"], refs["ref_len"], refs["ref_owner"], len(targets))
    text, text_len = encode_texts([corpus.texts[i] for i in targets], corpus.token, corpus.tokenizer)
    text, text_len = text.to(device), text_len.to(device)
    for name, cond, g in [("no style", None, 1.0), *((g, model.style_cond(mu), g) for g in guidances)]:
        generator = torch.Generator(device=device).manual_seed(seed)
        # FlowStudent.sample keeps the control points, which the encoder reads, rather than decoding them.
        written = FlowStudent.sample(model, text, text_len, 0.0, generator=generator, cond=cond, style_guidance=g)
        scores[name] = hit_rate(embed(model, [off.numpy() for off, _, _ in written], device))
    return scores, len(writers), len(targets)


def main(args):
    model, ckpt = load_model(args.checkpoint)
    if ckpt.get("architecture") != "style_spline_flow":
        raise SystemExit(f"{args.checkpoint} is not a style student")
    train_args = ckpt["args"]
    _, val = Corpus.split(train_args["corpus"], train_args["val_shards"])
    _, source_val = Corpus.split(val.meta["spline"]["source"], train_args["val_shards"])
    references = References(val, train_args["refs"], train_args["crop"], train_args["ref_points"])

    scores, n_writers, n_lines = style_match(model, val, references, args.guidance, args.lines_per_writer)
    print(f"style match over {n_lines} lines of {n_writers} held-out writers "
          f"(chance is about 1 in the writers of a source):")
    for name, score in scores.items():
        print(f"  {'guidance ' + str(name) if isinstance(name, float) else name:>14}: {score:.3f}")

    img = style_samples(model, val, source_val, references, args.sample_lines, guidances=args.guidance)
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "style_guidance.png")
    img.save(out)
    print(f"wrote {out}: per writer, real, no style, then guidance {args.guidance}")
    if args.wandb_project:
        run_id = open(os.path.join(os.path.dirname(args.checkpoint), "wandb_id")).read().strip()
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, id=run_id, resume="allow")
        run.log({"style_guidance": wandb.Image(img, caption=f"real, no style, guidance {args.guidance}"),
                 **{f"style_match/{k}": v for k, v in scores.items()}})
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--lines-per-writer", type=int, default=8)
    ap.add_argument("--sample-lines", type=int, default=8, help="held-out writers in the image")
    ap.add_argument("--out", help="default: style_guidance.png next to the checkpoint")
    ap.add_argument("--wandb-project")
    ap.add_argument("--wandb-entity")
    main(ap.parse_args())
