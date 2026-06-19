#!/usr/bin/env bash
# Decoding-temperature diagnostic on existing SFT checkpoints. Waits for any in-flight train/probe to finish
# (so it doesn't contend for the GPU), then sweeps temps on v3/v4/v5. No training.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=TORCH_SDPA
PY="$HOME/spark1/.venv/bin/python"
echo "=== waiting for GPU (v5 train/probe) to free ($(date +%T)) ==="
while pgrep -f 'sft_warmup|support_probe' >/dev/null 2>&1; do sleep 30; done
echo "=== GPU free; starting temp sweep ($(date +%T)) ==="
for v in v3 v4 v5; do
  [ -d "checkpoints/sft_rx_$v/final" ] || { echo "skip $v (no checkpoint)"; continue; }
  $PY -m framework.gates.temp_sweep --env tasks.rx_triage.env --model checkpoints/medgemma4b_text \
    --adapter "checkpoints/sft_rx_$v/final" --tag "$v" --temps 0.0,0.5,1.0 --n-cases 40 --k 8 \
    --out "results/temp_sweep_$v.json"
done
echo "=== TEMPSWEEP DONE ($(date +%T)) ==="
