#!/usr/bin/env bash
# Improved-SFT (curriculum) train + Gate-2 probe for rx_triage. Reuses the unchanged base probe (probe_base_t10).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=TORCH_SDPA
PY="$HOME/spark1/.venv/bin/python"
echo "=== TRAIN improved SFT ($(date +%T)) ==="
$PY -m framework.rl.sft_warmup --base checkpoints/medgemma4b_text --data data/rx_triage_sft_v2.jsonl \
    --out checkpoints/sft_rx_v2 --epochs 2 --batch 4 --lr 1e-4 --seed 1
echo "=== PROBE improved SFT ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.rx_triage.env --model checkpoints/medgemma4b_text \
    --adapter checkpoints/sft_rx_v2/final --tag sft_v2 --k 12 --n-cases 60 --max-turns 10 \
    --conditions hidden,anchor --out results/probe_sft_v2_t10.json
echo "=== ALLDONE ($(date +%T)) ==="
