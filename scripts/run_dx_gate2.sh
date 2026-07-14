#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=TORCH_SDPA
PY="$HOME/spark1/.venv/bin/python"
echo "=== [1/4] DX base probe ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.diagnostics.env --model checkpoints/medgemma4b_text \
    --tag dx_base --k 12 --n-cases 60 --max-turns 10 --conditions hidden,anchor --out results/probe_dx_base_t10.json
echo "=== [2/4] gen DX curriculum SFT data ($(date +%T)) ==="
$PY -m tasks.diagnostics.gen_data --mode curriculum_v3 --n 600 --out data/dx_sft.jsonl
echo "=== [3/4] DX SFT (v3-style: 600 traj, 2 ep) ($(date +%T)) ==="
$PY -m framework.rl.sft_warmup --base checkpoints/medgemma4b_text --data data/dx_sft.jsonl \
    --out checkpoints/dx_sft_v3 --epochs 2 --batch 4 --lr 1e-4 --seed 1
echo "=== [4/4] DX SFT probe ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.diagnostics.env --model checkpoints/medgemma4b_text \
    --adapter checkpoints/dx_sft_v3/final --tag dx_sft_v3 --k 12 --n-cases 60 --max-turns 10 \
    --conditions hidden,anchor --out results/probe_dx_sft_v3_t10.json
echo "=== DXDONE ($(date +%T)) ==="
