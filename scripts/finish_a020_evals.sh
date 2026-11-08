#!/usr/bin/env bash
# Finish a0.20's validation evals (the checkpoint-20->final probe is already running), then aggregate the sweep.
# Each probe is wrapped in `timeout` so a vLLM/LoRA hang (seen once) can't stall the whole thing.
set -uo pipefail
PY="$HOME/spark1/.venv/bin/python"; BASE=checkpoints/medgemma4b_text; export DX_VARIANT=deephedge
LOG=results/procreward_pilot/sweep; OUT=checkpoints/procreward/sweep_a0.20; T=2400

emeval () { "$PY" -m framework.procreward.eval_metrics --records "$1" --out-json "$2" > "${2%.json}.txt" 2>&1 || echo "eval fail $2"; }
probe () { # tag adapter k outjson
  timeout $T "$PY" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
    --adapter "$2" --tag "$1" --k "$3" --n-cases 120 --conditions hidden --max-turns 12 --seed 202 \
    --out "$4" > "${4%.json}_probe.log" 2>&1 || echo "probe fail/timeout $1"
  [ -f "${4%.json}.records.jsonl" ] && emeval "${4%.json}.records.jsonl" "${4%.json}_metrics.json"
}

echo "waiting for the running checkpoint-20 probe to produce records..."
for i in $(seq 1 80); do
  [ -f "$LOG/a0.20_final.records.jsonl" ] && break
  pgrep -cf support_probe >/dev/null 2>&1 || { echo "probe died before records"; break; }
  sleep 30
done
[ -f "$LOG/a0.20_final.records.jsonl" ] && emeval "$LOG/a0.20_final.records.jsonl" "$LOG/a0.20_final_metrics.json"

probe a0.20_checkpoint-10 "$OUT/checkpoint-10" 12 "$LOG/a0.20_checkpoint-10.json"
probe a0.20_greedy        "$OUT/checkpoint-20" 1  "$LOG/a0.20_greedy.json"

echo "=== a0.20 evals done -> aggregating sweep ==="
DX_VARIANT=deephedge PYTHONPATH="$PWD" "$PY" scripts/analyze_alpha_sweep.py > "$LOG/analyze_console.log" 2>&1 || echo "analyze failed"
echo "===== SWEEP FULLY DONE — see $LOG/alpha_sweep_report.md ====="
