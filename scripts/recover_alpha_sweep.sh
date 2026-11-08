#!/usr/bin/env bash
# Recover the crashed alpha runs. Root cause: reusing ONE node-B server across multiple trainings corrupts the
# NCCL weight-sync communicator (runs 2+ fail with "unhandled system error"). Fix: restart node B + clear node A
# BEFORE each training. Re-runs the alphas passed as args (e.g. 0.10 0.20). alpha=0.05 already completed.
set -uo pipefail
PY="$HOME/spark1/.venv/bin/python"
BASE=checkpoints/medgemma4b_text
export DX_VARIANT=deephedge
SH=192.168.100.2; SP=8001
VAL_SEED=202; STEPS=30; SAVE=10; NCASES=300; KP=120
OUT=checkpoints/procreward; LOG=results/procreward_pilot/sweep
source ~/.secrets/anthropic.env

restart_server () {
  # tmux-based (reliable; setsid-over-ssh was flaky). Full clean, fresh log, then health-only readiness check.
  ssh -o BatchMode=yes "$SH" 'tmux kill-server 2>/dev/null; pkill -9 -f "vllm-serve" 2>/dev/null; \
    pkill -9 -f "trl" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null; sleep 6; rm -f ~/vllm_serve.log; \
    tmux new-session -d -s prsrv "bash ~/serve_pr.sh > ~/vllm_serve.log 2>&1"' 2>/dev/null || true
  for i in $(seq 1 50); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://$SH:$SP/health 2>/dev/null)" = "307" ] && { echo "  node B ready (iter $i)"; sleep 3; return 0; }
    sleep 8
  done
  echo "  node B FAILED to become ready"; return 1
}

probe () { # tag adapter k outjson
  "$PY" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
    --adapter "$2" --tag "$1" --k "$3" --n-cases "$KP" --conditions hidden --max-turns 12 \
    --seed "$VAL_SEED" --out "$4" > "${4%.json}_probe.log" 2>&1 || echo "  [probe FAILED: $1]"
  "$PY" -m framework.procreward.eval_metrics --records "${4%.json}.records.jsonl" \
    --out-json "${4%.json}_metrics.json" > "${4%.json}_metrics.txt" 2>&1 || echo "  [eval FAILED: $1]"
}

for A in "$@"; do
  tag="a${A}"; echo "===== RECOVER ALPHA $A ====="
  pkill -9 -f "EngineCore" 2>/dev/null; pkill -9 -f "vllm" 2>/dev/null; sleep 10  # clear node A NCCL/GPU state
  restart_server || { echo "abort $A"; continue; }
  rm -rf "$OUT/sweep_${tag}"
  echo "  training..."
  "$PY" -m framework.rl.grpo_disagg --base "$BASE" --env tasks.diagnostics.env_rv \
    --server-host "$SH" --server-port "$SP" \
    --n-cases "$NCASES" --max-turns 12 --max-steps "$STEPS" --save-steps "$SAVE" \
    --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --seed 1 \
    --infogain-coef 0.0 --claude-alpha "$A" --claude-norm zscore_clip \
    --claude-cache "$LOG/judge_cache.jsonl" --claude-log "$LOG/${tag}_rewardlog.jsonl" \
    --out "$OUT/sweep_${tag}" > "$LOG/${tag}_train.log" 2>&1
  echo "  $(grep -a 'DONE' "$LOG/${tag}_train.log" | tail -1 || echo 'train produced no DONE line')"
  for CK in checkpoint-10 checkpoint-20 final; do
    [ -d "$OUT/sweep_${tag}/$CK" ] && probe "${tag}_${CK}" "$OUT/sweep_${tag}/$CK" 12 "$LOG/${tag}_${CK}.json"
  done
  [ -d "$OUT/sweep_${tag}/final" ] && probe "${tag}_greedy" "$OUT/sweep_${tag}/final" 1 "$LOG/${tag}_greedy.json"
done
echo "===== RECOVER DONE ====="
