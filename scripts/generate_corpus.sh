#!/usr/bin/env bash
# Generate teacher corpus shards on several GPUs at once, one job per GPU, then
# push the result to the W&B corpus artifact once (if --wandb-project is given).
#
#   scripts/generate_corpus.sh <gpus> <first_shard> <num_shards> [generate_teacher_corpus.py args ...]
#   scripts/generate_corpus.sh 0,1,3 0 64 --wandb-entity <entity> --wandb-project <project>
#
# The shard range is split into contiguous chunks, one per GPU. Each job logs to
# logs/generate_gpu<g>.log. Rerunning the same command skips finished shards.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <gpus, e.g. 0,1,3> <first_shard> <num_shards> [generate_teacher_corpus.py args ...]" >&2
    exit 2
fi
IFS=, read -ra GPUS <<< "$1"
FIRST="$2"
TOTAL="$3"
shift 3

# The jobs don't push themselves: pushes that finish together can drop each
# other's shards from the newest version, so the push happens once at the end.
GEN_ARGS=()
PUSH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb-entity|--wandb-project|--artifact)
            PUSH_ARGS+=("$1" "$2"); shift 2 ;;
        --out-dir)
            GEN_ARGS+=("$1" "$2"); PUSH_ARGS+=(--corpus "$2"); shift 2 ;;
        *)
            GEN_ARGS+=("$1"); shift ;;
    esac
done

# Sync .venv once, so the parallel jobs don't race over it.
"$ROOT_DIR/scripts/run_in_apptainer.sh" cpu true

mkdir -p "$ROOT_DIR/logs"
CHUNK=$(( (TOTAL + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
PIDS=()
for i in "${!GPUS[@]}"; do
    start=$(( FIRST + i * CHUNK ))
    count=$(( FIRST + TOTAL - start < CHUNK ? FIRST + TOTAL - start : CHUNK ))
    (( count > 0 )) || break
    gpu="${GPUS[$i]}"
    log="$ROOT_DIR/logs/generate_gpu${gpu}.log"
    echo "GPU $gpu: shards $start-$(( start + count - 1 )), log $log"
    INK_NO_SYNC=1 "$ROOT_DIR/scripts/run_in_apptainer.sh" "$gpu" python scripts/generate_teacher_corpus.py \
        --first-shard "$start" --num-shards "$count" ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} > "$log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=1
done
if (( FAILED )); then
    echo "a generation job failed; see logs/generate_gpu*.log (rerun to resume)" >&2
fi

# Finished shards are complete files even if a job failed, so push them anyway.
if [[ " ${PUSH_ARGS[*]-} " == *" --wandb-project "* ]]; then
    INK_NO_SYNC=1 "$ROOT_DIR/scripts/run_in_apptainer.sh" cpu python scripts/corpus_artifact.py push "${PUSH_ARGS[@]}"
else
    echo "no --wandb-project given, so nothing was pushed to W&B"
fi
exit "$FAILED"
