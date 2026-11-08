#!/usr/bin/env bash
# Finish the alpha sweep after a0.10 was launched directly. Root cause of the crashes: a vLLM-server RESTART (or
# an eval->train transition without the GPU fully released) corrupts the NCCL weight-sync; only a FIRST init
# against a fully-ready fresh server with node A's GPU free works. This script encodes that:
#   wait a0.10 train -> eval a0.10 -> (free node A GPU) -> FRESH node B server -> a0.20 train DIRECT -> eval a0.20.
set -uo pipefail
PY="$HOME/spark1/.venv/bin/python"
BASE=checkpoints/medgemma4b_text
export DX_VARIANT=deephedge
SH=192.168.100.2; SP=8001
VAL_SEED=202; STEPS=30; SAVE=10; NCASES=300; KP=120
OUT=checkpoints/procreward; LOG=results/procreward_pilot/sweep
source ~/.secrets/anthropic.env

wait_a_gpu_free () {  # ensure node A GPU fully released before an NCCL init
  for i in $(seq 1 30); do
    pkill -9 -f "EngineCore" 2>/dev/null; pkill -9 -f "vllm" 2>/dev/null
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "${n:-1}" = "0" ] && { echo "  node A GPU free"; sleep 3; return 0; }
    sleep 6
  done
  echo "  node A GPU still busy"; return 0
}

fresh_server_B () {  # full clean + fresh tmux server + wait for real readiness + buffer
  ssh -o BatchMode=yes "$SH" 'tmux kill-server 2>/dev/null; pkill -9 -f vllm 2>/dev/null; pkill -9 -f trl 2>/dev/null; sleep 6; rm -f ~/vllm_serve.log; tmux new-session -d -s prsrv "bash ~/serve_pr.sh > ~/vllm_serve.log 2>&1"' 2>/dev/null || true
  for i in $(seq 1 50); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://$SH:$SP/health 2>/dev/null)" = "307" ] && { echo "  node B ready (iter $i)"; sleep 20; return 0; }  # +20s buffer for CUDA-graph capture
    sleep 8
  done
  echo "  node B FAILED"; return 1
}

probe () { # tag adapter k outjson
  "$PY" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
    --adapter "$2" --tag "$1" --k "$3" --n-cases "$KP" --conditions hidden --max-turns 12 \
    --seed "$VAL_SEED" --out "$4" > "${4%.json}_probe.log" 2>&1 || echo "  [probe FAILED: $1]"
  "$PY" -m framework.procreward.eval_metrics --records "${4%.json}.records.jsonl" \
    --out-json "${4%.json}_metrics.json" > "${4%.json}_metrics.txt" 2>&1 || echo "  [eval FAILED: $1]"
}

eval_alpha () { # tag
  local tag="$1"
  for CK in checkpoint-10 checkpoint-20 final; do
    [ -d "$OUT/sweep_${tag}/$CK" ] && probe "${tag}_${CK}" "$OUT/sweep_${tag}/$CK" 12 "$LOG/${tag}_${CK}.json"
  done
  [ -d "$OUT/sweep_${tag}/final" ] && probe "${tag}_greedy" "$OUT/sweep_${tag}/final" 1 "$LOG/${tag}_greedy.json"
}

train_alpha_direct () { # alpha  (assumes fresh server already up + node A GPU free)
  local A="$1"; local tag="a${A}"
  rm -rf "$OUT/sweep_${tag}"
  "$PY" -m framework.rl.grpo_disagg --base "$BASE" --env tasks.diagnostics.env_rv \
    --server-host "$SH" --server-port "$SP" \
    --n-cases "$NCASES" --max-turns 12 --max-steps "$STEPS" --save-steps "$SAVE" \
    --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --seed 1 \
    --infogain-coef 0.0 --claude-alpha "$A" --claude-norm zscore_clip \
    --claude-cache "$LOG/judge_cache.jsonl" --claude-log "$LOG/${tag}_rewardlog.jsonl" \
    --out "$OUT/sweep_${tag}" > "$LOG/${tag}_train.log" 2>&1
}

echo "===== waiting for a0.10 training to finish ====="
while pgrep -f "claude-alpha 0.10" >/dev/null || { ! grep -qa 'DONE steps=' "$LOG/a0.10_train.log" 2>/dev/null && pgrep -f grpo_disagg >/dev/null; }; do sleep 60; done
echo "a0.10 train: $(grep -a 'DONE steps=' "$LOG/a0.10_train.log" | tail -1 || echo 'NO DONE (may have crashed)')"

echo "===== eval a0.10 ====="
wait_a_gpu_free
eval_alpha a0.10

echo "===== a0.20: fresh server + direct train ====="
wait_a_gpu_free
fresh_server_B || { echo "abort a0.20"; exit 1; }
train_alpha_direct 0.20
echo "a0.20 train: $(grep -a 'DONE steps=' "$LOG/a0.20_train.log" | tail -1 || echo 'NO DONE (crashed)')"
wait_a_gpu_free
eval_alpha a0.20
echo "===== SWEEP FINISH DONE ====="
