#!/usr/bin/env bash
# Train an SFT variant + Gate-2 probe (reuses unchanged base probe). Env: DATA, OUT, TAG, OUTJSON.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ATTENTION_BACKEND=TORCH_SDPA
PY="$HOME/spark1/.venv/bin/python"
: "${DATA:?}"; : "${OUT:?}"; : "${TAG:?}"; : "${OUTJSON:?}"
echo "=== TRAIN $TAG ($(date +%T)) ==="
$PY -m framework.rl.sft_warmup --base checkpoints/medgemma4b_text --data "$DATA" --out "$OUT" \
    --epochs 2 --batch 4 --lr 1e-4 --seed 1
echo "=== PROBE $TAG ($(date +%T)) ==="
$PY -m framework.gates.support_probe --env tasks.rx_triage.env --model checkpoints/medgemma4b_text \
    --adapter "$OUT/final" --tag "$TAG" --k 12 --n-cases 60 --max-turns 10 --conditions hidden,anchor --out "$OUTJSON"
echo "=== ALLDONE ($(date +%T)) ==="
