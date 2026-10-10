# Running jobs on Slurm

`scripts/submit_slurm.sh` submits any command as a single-node batch job that
runs from the repository root with your current shell environment. It does not
know about any particular job, so the same helper covers training runs, bulk
synthetic data generation and benchmarks.

## Before the first job

Set up the environment on the login node, not inside the job:

```bash
git clone <this repo> && cd Ink_Tokenizer
uv sync --all-extras
```

Keep the checkout and `UV_CACHE_DIR` off a small-quota `$HOME`, because the
CUDA wheels alone take several GB.

## Submitting

Check what will be submitted first:

```bash
scripts/submit_slurm.sh --dry-run -- uv run --no-sync python -c \
    "import torch; print(torch.cuda.get_device_name())"
```

Then drop `--dry-run`:

```bash
scripts/submit_slurm.sh --account <alloc> --job-name mamba-ink --time 12:00:00 -- \
    uv run --no-sync python <training-script> [args...]
```

Everything after `--` is the command, passed through exactly as typed.
`scripts/submit_slurm.sh --help` lists the options. The defaults are partition
`rtx-small`, 24 h wall time, no modules, and logs in `logs/slurm/%x-%j.out`
(job name and job id, stdout and stderr combined).

- **Always `uv run --no-sync`.** A plain `uv run` re-syncs `.venv` when the job
  starts. That needs network access from the compute node, and two queued jobs
  starting together will race over the same environment.
- **The job runs the tree as it is when the job starts**, not as it was at
  submission. Avoid editing a checkout that has jobs queued against it. The log
  header records `git describe --dirty`, so a checkpoint can be traced back to
  the code that produced it.
- **Check the log header before trusting a long run.** It lists the GPUs from
  `nvidia-smi -L`, or warns when there is none. Finding out after an overnight
  job that it ran on CPU is the failure this header is there to catch.
- **GPUs come from the partition.** GPU/GRES options, and every other setting
  the helper manages, are rejected in `--sbatch-option`. For multi-GPU on one
  node, use a launcher as the command, e.g.
  `uv run --no-sync torchrun --standalone --nproc-per-node=<n> <script>`.
- **Other sbatch settings** take one token each:
  `--sbatch-option=--mail-type=END --sbatch-option=--mail-user=<you>`.
  `--sbatch-option=--dependency=afterok:<jobid>` chains a resumed run behind
  one that is about to hit the wall-time limit.
- **Modules** are opt-in and repeatable (`--module cuda/12.2`).
