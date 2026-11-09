#!/bin/bash
# Read-only check: V1's own support_probe (k=12, held-out seed 100, hidden, env_rv deephedge), with only the stop
# configuration changed through a symlinked model view. V1 checkpoints, results and code are untouched; outputs go to
# the smoke logs. "control" repeats plain GRPO with the original config, to test whether today's stack reproduces V1.
set -u
cd "$HOME/rl-gains-in-clinical-tasks"
O=V2/benchmark_smoke_logs/stop_token/support_probe; mkdir -p "$O"; PY="$HOME/spark1/.venv/bin/python"
FIX="$HOME/v2_bench/medgemma4b_stopfix"; ORIG=checkpoints/medgemma4b_text; export DX_VARIANT=deephedge
run(){ # tag model adapter
  local s=$(date +%s); local a=(); [ -n "$3" ] && a=(--adapter "$3")
  timeout 3600 "$PY" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$2" "${a[@]}" --tag "$1" \
    --k 12 --n-cases 60 --conditions hidden --max-turns 12 --seed 100 --out "$O/$1.json" > "$O/$1.stdout" 2>&1
  echo "$1 exit $? secs $(( $(date +%s)-s ))" >> "$O/progress.txt"; sleep 8
}
run stop_plain_s1  "$FIX"  checkpoints/dh_grpo_base/final
run stop_credit_s1 "$FIX"  checkpoints/dh_grpo_infogain/final
run stop_base      "$FIX"  ""
run ctrl_nostop_plain_s1 "$ORIG" checkpoints/dh_grpo_base/final
echo finished >> "$O/progress.txt"
