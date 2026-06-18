#!/usr/bin/env bash
# Run ON the rollout node (Spark-B) inside a detached tmux (started by the GRPO orchestrator).
# $1 = absolute model dir to serve. $2 = port (default 8000). Logs to ~/vllm_serve.log (grep for readiness).
# tmux new-session -d returns immediately, so the launching ssh returns in ~1s (no hung channel).
set -u
PORT="${2:-8000}"
cd "$HOME" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
exec "$HOME/spark1/.venv/bin/trl" vllm-serve --model "$1" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization 0.55 --dtype bfloat16 --max-model-len 4096 > "$HOME/vllm_serve.log" 2>&1
