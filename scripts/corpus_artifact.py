"""Keep a teacher corpus directory (scripts/generate_teacher_corpus.py) in a
W&B dataset artifact, so shards generated on different clusters build one
corpus, and a training run records exactly which version it used.

push  adds the local files that the latest version lacks (meta.json,
      lines.txt and shards, matched by name) as a new version. Files already
      in it are not uploaded again.
pull  downloads a version's files that are missing locally.

Each push builds on the version that was latest when it started, so two
pushes that overlap can drop each other's new shards from the newest
version. The shards are still on disk, and pushing again from either
directory adds them back.

Usage: python scripts/corpus_artifact.py {push,pull} --wandb-project P [--wandb-entity E]
           [--artifact teacher_corpus[:version]] [--corpus data/teacher_corpus]
"""
import argparse
import glob
import os

import wandb

ARTIFACT_TYPE = "dataset"


def add_wandb_args(ap, artifact_default="teacher_corpus", artifact_help="corpus artifact name, optionally :version"):
    ap.add_argument("--wandb-entity", default=None, help="W&B user or team (default: your W&B default entity)")
    ap.add_argument("--wandb-project", default=None, help="W&B project; W&B is not used without it")
    ap.add_argument("--artifact", default=artifact_default, help=artifact_help)


def artifact_path(entity, project, artifact):
    path = f"{project}/{artifact}" if entity is None else f"{entity}/{project}/{artifact}"
    return path if ":" in artifact else path + ":latest"


def shard_names(artifact):
    return sorted(name for name in artifact.manifest.entries if name.startswith("shard_"))


def push(corpus_dir, entity, project, artifact_name):
    name = artifact_name.split(":")[0]
    api = wandb.Api()
    try:
        exists = api.artifact_exists(artifact_path(entity, project, name))
    except ValueError:  # raised instead of returning False while the project doesn't exist yet
        exists = False
    if exists:
        artifact = api.artifact(artifact_path(entity, project, name)).new_draft()
    else:
        artifact = wandb.Artifact(name, ARTIFACT_TYPE)

    local = ["meta.json", "lines.txt"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(corpus_dir, "shard_*.npz")))
    new = [f for f in local if f not in artifact.manifest.entries]
    if not new:
        print(f"{name}: nothing new to push", flush=True)
        return
    # Shards are written once and never modified, so wandb can read them in
    # place instead of copying gigabytes into its staging cache.
    for f in new:
        artifact.add_file(os.path.join(corpus_dir, f), name=f, policy="immutable", skip_cache=True)
    artifact.metadata["shards"] = len(shard_names(artifact))
    with wandb.init(entity=entity, project=project, job_type="push-corpus") as run:
        run.log_artifact(artifact)
        artifact.wait()
    print(f"pushed {len(new)} files as {artifact.name} ({artifact.metadata['shards']} shards)", flush=True)


def pull(corpus_dir, entity, project, artifact_name, run=None):
    """Returns the artifact; with a run, it is recorded as that run's input."""
    path = artifact_path(entity, project, artifact_name)
    artifact = run.use_artifact(path) if run else wandb.Api().artifact(path)
    artifact.download(root=corpus_dir, skip_cache=True)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--corpus", default="data/teacher_corpus")
    add_wandb_args(ap)
    args = ap.parse_args()
    if args.wandb_project is None:
        ap.error("--wandb-project is required")
    if args.action == "push":
        push(args.corpus, args.wandb_entity, args.wandb_project, args.artifact)
    else:
        print(f"pulled {pull(args.corpus, args.wandb_entity, args.wandb_project, args.artifact).name} into {args.corpus}")
