"""Render a flow student checkpoint's sample image with a chosen number of
Euler steps and upload it to W&B.

The image is train_flow.py's: rows alternate held-out corpus ink and the
student writing the same line. It goes to a new W&B run (job type
"visualize") rather than the training run, which may still be logging. For a
spline flow student, pass its source corpus as --corpus, so the corpus rows
are the original ink rather than control points.

Usage: python scripts/visualize_flow.py --checkpoint models/student/<run>/checkpoint.pt
           --wandb-project P [--wandb-entity E] [--steps 16] [--guidance 2.0] [--font-guidance W]
           [--lines 8] [--corpus DIR] [--artifact NAME|none]
"""
import argparse
import os

import wandb

from corpus_artifact import add_wandb_args, pull, shard_names
from ink_corpus import Corpus
from sample_student import load_model, teacher_vs_student


def main(args):
    model, ckpt = load_model(args.checkpoint)
    if ckpt.get("architecture") not in ("flow", "spline_flow"):
        raise SystemExit(f"{args.checkpoint} is not a flow student")
    train_args = ckpt["args"]
    corpus_dir = args.corpus or train_args["corpus"]
    artifact = args.artifact or train_args.get("artifact", "none")
    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))

    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, job_type="visualize",
                     name=f"{run_name}-euler{args.steps}"
                          + (f"-font{args.font_guidance:g}" if args.font_guidance != 1 else ""),
                     config={**vars(args), "checkpoint_step": ckpt["step"], "corpus": corpus_dir})
    names = None
    if artifact != "none":
        names = shard_names(pull(corpus_dir, args.wandb_entity, args.wandb_project, artifact, run))
    _, val = Corpus.split(corpus_dir, train_args.get("val_shards", 1), names)

    model.sample_steps = args.steps
    if args.guidance is not None:
        model.sample_guidance = args.guidance
    if args.font_guidance != 1:
        model.guide_from_fonts(ckpt["charset"], args.font_guidance)
    img = teacher_vs_student(model, val, args.lines, args.seed)
    if args.out:
        img.save(args.out)
        print(f"saved {args.out}", flush=True)
    caption = (f"{run_name} step {ckpt['step']}, {args.steps} Euler steps, guidance {model.sample_guidance}, "
               f"font guidance {args.font_guidance:g}; "
               f"rows alternate corpus ink, flow student")
    run.log({"samples": wandb.Image(img, caption=caption)})
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=16, help="Euler steps")
    ap.add_argument("--guidance", type=float, help="classifier-free guidance scale (default: the checkpoint's)")
    ap.add_argument("--font-guidance", type=float, default=1.0,
                    help="weight w guiding away from font ink (models trained on a font corpus; 1 = off)")
    ap.add_argument("--lines", type=int, default=8, help="held-out lines drawn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", help="corpus directory (default: the one the checkpoint was trained on)")
    ap.add_argument("--out", help="also save the image here")
    add_wandb_args(ap, artifact_default=None,
                   artifact_help="corpus artifact (default: the version the checkpoint was trained on); "
                                 "none uses --corpus as it is")
    args = ap.parse_args()
    if args.wandb_project is None:
        ap.error("--wandb-project is required")
    main(args)
