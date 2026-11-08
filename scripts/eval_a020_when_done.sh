#!/usr/bin/env bash
# Wait for a0.20 training to finish, then run its 4 validation evals (val seed 202) on node A (in-process vLLM;
# no node B server restart needed for eval). After this, the sweep is complete -> run analyze_alpha_sweep.py.
set -uo pipefail
PY="$HOME/spark1/.venv/bin/python"
BASE=checkpoints/medgemma4b_text
export DX_VARIANT=deephedge
VAL_SEED=202; KP=120
OUT=checkpoints/procreward; LOG=results/procreward_pilot/sweep

echo "waiting for a0.20 training DONE..."
while ! grep -qa 'DONE steps=' "$LOG/a0.20_train.log" 2>/dev/null; do
  pgrep -f "claude-alpha 0.20" >/dev/null || pgrep -f grpo_disagg >/dev/null || { echo "grpo gone"; break; }
  sleep 60
done
echo "a0.20 train: $(grep -a 'DONE steps=' "$LOG/a0.20_train.log" | tail -1 || echo 'NO DONE')"

# free node A GPU before eval probes
for i in $(seq 1 20); do
  pkill -9 -f "EngineCore" 2>/dev/null; pkill -9 -f "vllm" 2>/dev/null
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" = "0" ] && break
  sleep 6
done

probe () { # tag adapter k outjson
  "$PY" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
    --adapter "$2" --tag "$1" --k "$3" --n-cases "$KP" --conditions hidden --max-turns 12 \
    --seed "$VAL_SEED" --out "$4" > "${4%.json}_probe.log" 2>&1 || echo "  [probe FAILED: $1]"
  "$PY" -m framework.procreward.eval_metrics --records "${4%.json}.records.jsonl" \
    --out-json "${4%.json}_metrics.json" > "${4%.json}_metrics.txt" 2>&1 || echo "  [eval FAILED: $1]"
}
for CK in checkpoint-10 checkpoint-20 final; do
  [ -d "$OUT/sweep_a0.20/$CK" ] && probe "a0.20_${CK}" "$OUT/sweep_a0.20/$CK" 12 "$LOG/a0.20_${CK}.json"
done
[ -d "$OUT/sweep_a0.20/final" ] && probe "a0.20_greedy" "$OUT/sweep_a0.20/final" 1 "$LOG/a0.20_greedy.json"
echo "===== A0.20 EVAL DONE — sweep complete ====="
