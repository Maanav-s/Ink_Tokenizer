"""Sample the teacher (scripts/teacher_model.py) over a list of lines and save
the results as .npz shards for training a student.

Lines are written unprimed, so every line is in a fresh random style. Each
batch gets its own bias, drawn uniformly from --bias-range, so the corpus
spans neat and messy handwriting. Lines that clean_sample() rejects are
dropped and counted.

Shard i draws its lines and seeds from i alone, so shards can be generated in
any order, by any number of jobs (--first-shard/--num-shards), and an
interrupted run resumes by skipping shards that already exist.

Each shard holds, for its kept lines:
  offsets  float32 (total points, 3)  normalized (dx, dy, end_of_stroke)
  chars    int16   (total points,)    index into the line of each point's char
  lengths  int32   (lines,)           points per line; split offsets/chars with it
  texts    str     (lines,)
  bias     float32 (lines,)
  rejected int32   ()                 lines the teacher failed, for comparison
meta.json beside the shards has the teacher's charset and normalization.

Usage: python scripts/generate_teacher_corpus.py [--lines data/teacher_corpus/lines.txt]
           [--out-dir data/teacher_corpus] [--first-shard 0] [--num-shards 1]
           [--lines-per-shard 2048] [--batch 128] [--device cuda]
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from teacher_model import CHECKPOINT, Teacher, default_checkpoint_dir
from text_to_ink import clean_sample


def generate_shard(teacher, lines, index, lines_per_shard, batch, bias_range):
    r = random.Random(index)
    chosen = sorted(r.sample(lines, lines_per_shard), key=len)  # similar lengths batch together
    mu, std = teacher.mu.numpy(), teacher.std.numpy()
    kept = {"offsets": [], "chars": [], "texts": [], "bias": []}
    rejected = 0
    for b in range(0, len(chosen), batch):
        texts = chosen[b:b + batch]
        bias = r.uniform(*bias_range)
        samples = teacher.sample(texts, bias=bias, seed=index * 100003 + b)
        for text, s in zip(texts, samples):
            offsets, chars, ok = clean_sample(text, s["offsets"].numpy(), s["char"].numpy(), s["finished"], mu, std)
            if not ok:
                rejected += 1
                continue
            kept["offsets"].append(offsets)
            kept["chars"].append(chars)
            kept["texts"].append(text)
            kept["bias"].append(bias)
    return {
        "offsets": np.concatenate(kept["offsets"]).astype(np.float32),
        "chars": np.concatenate(kept["chars"]).astype(np.int16),
        "lengths": np.array([len(o) for o in kept["offsets"]], dtype=np.int32),
        "texts": np.array(kept["texts"]),
        "bias": np.array(kept["bias"], dtype=np.float32),
        "rejected": np.int32(rejected),
    }, rejected


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    teacher = Teacher(args.checkpoint, args.device)
    lines = [ln for ln in open(args.lines).read().split("\n") if ln]
    meta_path = os.path.join(args.out_dir, "meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({"charset": teacher.charset, "mu": teacher.mu.tolist(), "std": teacher.std.tolist(),
                       "teacher": CHECKPOINT, "lines": os.path.abspath(args.lines),
                       "bias_range": args.bias_range}, f, indent=1)

    for index in range(args.first_shard, args.first_shard + args.num_shards):
        path = os.path.join(args.out_dir, f"shard_{index:05d}.npz")
        if os.path.exists(path):
            continue
        start = time.time()
        shard, rejected = generate_shard(teacher, lines, index, args.lines_per_shard, args.batch, args.bias_range)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **shard)
        os.replace(tmp, path)  # a killed job never leaves a half-written shard
        print(f"{path}: kept {len(shard['texts'])}, rejected {rejected}, "
              f"{len(shard['offsets'])} points, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lines", default="data/teacher_corpus/lines.txt")
    ap.add_argument("--out-dir", default="data/teacher_corpus")
    ap.add_argument("--first-shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--lines-per-shard", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bias-range", type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument("--checkpoint", default=default_checkpoint_dir())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
