"""Move student checkpoints between machines through W&B model artifacts.

Push from a cluster and pull onto the workstation to sample from them.

push  uploads <run-dir>/checkpoint.pt and metrics.jsonl as a new version of
      the artifact (default: student-<run-dir name>), aliased step-<step>.
pull  downloads a version into --run-dir, overwriting the files there.

wandb_id is not uploaded, so training in a pulled directory starts its own
W&B run rather than resuming the cluster's.

Usage: python scripts/checkpoint_artifact.py {push,pull} --run-dir models/student/<name>
           --wandb-project P [--wandb-entity E] [--artifact student-<name>[:version|:step-N]]
"""
import argparse
import os

import torch
import wandb

from corpus_artifact import add_wandb_args, artifact_path

ARTIFACT_TYPE = "model"
FILES = ["checkpoint.pt", "metrics.jsonl"]


def default_name(run_dir):
    return "student-" + os.path.basename(os.path.normpath(run_dir))


def push(run_dir, entity, project, artifact_name):
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    artifact = wandb.Artifact(artifact_name.split(":")[0], ARTIFACT_TYPE,
                              metadata={"step": ckpt["step"], "config": ckpt["config"],
                                        "corpus": ckpt["args"].get("artifact")})
    for f in FILES:
        if os.path.exists(os.path.join(run_dir, f)):
            artifact.add_file(os.path.join(run_dir, f), name=f)
    with wandb.init(entity=entity, project=project, job_type="push-checkpoint") as run:
        run.log_artifact(artifact, aliases=["latest", f"step-{ckpt['step']}"])
        artifact.wait()
    print(f"pushed {run_dir} at step {ckpt['step']} as {artifact.name}", flush=True)


def pull(run_dir, entity, project, artifact_name):
    artifact = wandb.Api().artifact(artifact_path(entity, project, artifact_name))
    artifact.download(root=run_dir, skip_cache=True)
    print(f"pulled {artifact.name} (step {artifact.metadata.get('step')}) into {run_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--run-dir", required=True)
    add_wandb_args(ap, artifact_default=None,
                   artifact_help="checkpoint artifact name, optionally :version or :step-N (default: student-<run-dir name>)")
    args = ap.parse_args()
    if args.wandb_project is None:
        ap.error("--wandb-project is required")
    name = args.artifact or default_name(args.run_dir)
    if args.action == "push":
        push(args.run_dir, args.wandb_entity, args.wandb_project, name)
    else:
        pull(args.run_dir, args.wandb_entity, args.wandb_project, name)
