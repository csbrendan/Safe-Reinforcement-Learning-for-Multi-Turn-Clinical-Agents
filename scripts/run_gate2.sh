#!/usr/bin/env bash
# Gate 2 (latent support) driver for rx_triage — base probe -> smoke SFT -> SFT probe.
# Sequential on one GPU (each stage exits before the next). Wrap in tmux for long runs.
#   tmux new -s gate2 -d "bash scripts/run_gate2.sh 2>&1 | tee logs_gate2.log"
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
PY="$HOME/spark1/.venv/bin/python"

BASE="${BASE:-checkpoints/medgemma4b_text}"
DATA="${DATA:-data/rx_triage_sft_smoke.jsonl}"
SFT_OUT="${SFT_OUT:-checkpoints/sft_rx_smoke}"
K="${K:-12}"; NC="${NC:-60}"
mkdir -p results

echo "=== [1/3] BASE support probe ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.rx_triage.env --model "$BASE" \
    --tag base --k "$K" --n-cases "$NC" --conditions hidden,anchor --out results/probe_base.json

echo "=== [2/3] smoke SFT ($(date +%T)) ==="
$PY -m framework.rl.sft_warmup --base "$BASE" --data "$DATA" --out "$SFT_OUT" \
    --epochs 2 --batch 4 --lr 1e-4 --seed 1

echo "=== [3/3] SFT support probe ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.rx_triage.env --model "$BASE" \
    --adapter "$SFT_OUT/final" --tag sft --k "$K" --n-cases "$NC" --conditions hidden,anchor \
    --out results/probe_sft.json

echo "=== Gate-2 report ($(date +%T)) ==="
$PY -m framework.gates.gate2_report --run base=results/probe_base.json --run sft=results/probe_sft.json \
    --cond hidden --k "$K"
echo "=== DONE ($(date +%T)) ==="
