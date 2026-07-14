#!/usr/bin/env bash
# Validation-only Claude process-reward ALPHA SWEEP (deephedge, 2-node).
# alpha in {0.05, 0.10, 0.20}; identical init/cases/group/prompts/steps/decoding; VALIDATION seed 202 ONLY.
# TEST seed 100 is never generated here. Trains each alpha, saves checkpoints, evaluates each checkpoint +
# greedy pass@1 on val seed 202. Node B (192.168.100.2:8001) must be serving.
set -uo pipefail
PY="$HOME/spark1/.venv/bin/python"
BASE=checkpoints/medgemma4b_text
export DX_VARIANT=deephedge
SH="${SERVER_HOST:-192.168.100.2}"; SP="${SERVER_PORT:-8001}"
VAL_SEED=202; STEPS=30; SAVE=10; NCASES=300; KP=120
OUT=checkpoints/procreward; LOG=results/procreward_pilot/sweep
mkdir -p "$OUT" "$LOG"
source ~/.secrets/anthropic.env

probe () { # tag adapter k outjson
  "$PY" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
    --adapter "$2" --tag "$1" --k "$3" --n-cases "$KP" --conditions hidden --max-turns 12 \
    --seed "$VAL_SEED" --out "$4" > "${4%.json}_probe.log" 2>&1 || echo "  [probe FAILED: $1]"
  "$PY" -m framework.procreward.eval_metrics --records "${4%.json}.records.jsonl" \
    --out-json "${4%.json}_metrics.json" > "${4%.json}_metrics.txt" 2>&1 || echo "  [eval FAILED: $1]"
}

for A in 0.05 0.10 0.20; do
  tag="a${A}"; echo "===== ALPHA $A : TRAIN ($STEPS steps) ====="
  "$PY" -m framework.rl.grpo_disagg --base "$BASE" --env tasks.diagnostics.env_rv \
    --server-host "$SH" --server-port "$SP" \
    --n-cases "$NCASES" --max-turns 12 --max-steps "$STEPS" --save-steps "$SAVE" \
    --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --seed 1 \
    --infogain-coef 0.0 --claude-alpha "$A" --claude-norm zscore_clip \
    --claude-cache "$LOG/judge_cache.jsonl" --claude-log "$LOG/${tag}_rewardlog.jsonl" \
    --out "$OUT/sweep_${tag}" > "$LOG/${tag}_train.log" 2>&1
  echo "  train done: $(grep -a 'DONE' "$LOG/${tag}_train.log" | tail -1)"
  echo "===== ALPHA $A : EVAL on val seed $VAL_SEED ====="
  for CK in checkpoint-10 checkpoint-20 final; do
    [ -d "$OUT/sweep_${tag}/$CK" ] && probe "${tag}_${CK}" "$OUT/sweep_${tag}/$CK" 12 "$LOG/${tag}_${CK}.json"
  done
  probe "${tag}_greedy" "$OUT/sweep_${tag}/final" 1 "$LOG/${tag}_greedy.json"
done
echo "===== SWEEP DONE ====="
