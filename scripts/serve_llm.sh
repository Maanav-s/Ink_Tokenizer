#!/usr/bin/env bash
# Download (on first run) and serve an open-weights chat model with vLLM, for
# scripts/demo_server.py --llm-url http://localhost:$PORT/v1.
#
#   CUDA_VISIBLE_DEVICES=0 scripts/serve_llm.sh
#   MODEL=google/gemma-4-E4B-it PORT=8001 GPU_MEM=0.8 scripts/serve_llm.sh [extra vllm serve args...]
#   MODEL=Qwen/Qwen3.5-4B scripts/serve_llm.sh --reasoning-parser qwen3
#
# vLLM pins its own torch, which clashes with the `model` extra's, so it runs
# from a uv tool environment rather than the project's .venv. Gemma is gated
# on Hugging Face: accept its license there and set HF_TOKEN.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-google/gemma-4-E4B-it}
PORT=${PORT:-8001}
# The fraction of the GPU vLLM takes. Gemma's E models hold about twice their
# "effective" parameters, so half a card leaves no room for the KV cache; the
# rest is left for demo_server.py's recognizer and student.
GPU_MEM=${GPU_MEM:-0.8}
export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}

# Bound to localhost so the server isn't open to the rest of the cluster;
# reach it through an SSH tunnel.
exec uvx --python 3.12 --from vllm vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --gpu-memory-utilization "$GPU_MEM" \
    "$@"
