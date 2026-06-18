#!/usr/bin/env bash
# Gate 2 (latent support) driver for rx_triage — base probe -> [smoke SFT] -> SFT probe -> report.
# Sequential on one GPU (each stage exits before the next). Wrap in tmux for long runs.
#   tmux new -s gate2 -d "bash scripts/run_gate2.sh 2>&1 | tee logs_gate2.log"
# Env knobs: MAXTURNS (dialogue horizon, default 10), SUFFIX (output tag), SKIP_SFT=1 (reuse existing adapter).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
PY="$HOME/spark1/.venv/bin/python"

BASE="${BASE:-checkpoints/medgemma4b_text}"
DATA="${DATA:-data/rx_triage_sft_smoke.jsonl}"
SFT_OUT="${SFT_OUT:-checkpoints/sft_rx_smoke}"
K="${K:-12}"; NC="${NC:-60}"; MAXTURNS="${MAXTURNS:-10}"; SUFFIX="${SUFFIX:-}"
SKIP_SFT="${SKIP_SFT:-0}"
B="results/probe_base${SUFFIX}.json"; S="results/probe_sft${SUFFIX}.json"
mkdir -p results

echo "=== [1/3] BASE support probe (max_turns=$MAXTURNS) ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.rx_triage.env --model "$BASE" \
    --tag base --k "$K" --n-cases "$NC" --max-turns "$MAXTURNS" --conditions hidden,anchor --out "$B"

if [ "$SKIP_SFT" = "1" ] && [ -d "$SFT_OUT/final" ]; then
  echo "=== [2/3] smoke SFT SKIPPED — reusing $SFT_OUT/final (data unchanged) ($(date +%T)) ==="
else
  echo "=== [2/3] smoke SFT ($(date +%T)) ==="
  $PY -m framework.rl.sft_warmup --base "$BASE" --data "$DATA" --out "$SFT_OUT" \
      --epochs 2 --batch 4 --lr 1e-4 --seed 1
fi

echo "=== [3/3] SFT support probe (max_turns=$MAXTURNS) ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.rx_triage.env --model "$BASE" \
    --adapter "$SFT_OUT/final" --tag sft --k "$K" --n-cases "$NC" --max-turns "$MAXTURNS" \
    --conditions hidden,anchor --out "$S"

echo "=== Gate-2 report ($(date +%T)) ==="
$PY -m framework.gates.gate2_report --run "base=$B" --run "sft=$S" --cond hidden --k "$K"
echo "=== DONE ($(date +%T)) ==="
