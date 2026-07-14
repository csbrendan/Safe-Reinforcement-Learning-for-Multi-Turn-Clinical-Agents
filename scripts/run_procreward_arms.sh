#!/usr/bin/env bash
# =============================================================================
# Claude process-reward GRPO — three-arm comparison on diagnostics deephedge.
# PREPARED, NOT LAUNCHED. This script refuses to run unless you pass --confirm,
# so it cannot be triggered by accident. Do NOT launch the full multi-seed GPU
# run without explicit approval.
#
#   plain GRPO            : analytic terminal reward only        (--infogain-coef 0 --claude-alpha 0)
#   info-gain GRPO        : analytic + lambda * per-ask infogain (--infogain-coef 0.2)
#   Claude process-reward : analytic + alpha * norm(Claude)      (--claude-alpha $ALPHA), 3 seeds
#
# Identical initialization, data, rollout settings, and seeds across arms.
# Validation set (seed 202) selects alpha + checkpoint. TEST set (seed 100) is
# untouched until the arms are frozen. support@12 is a diagnostic only.
# =============================================================================
set -euo pipefail

CONFIRM="${1:-}"
BASE="${BASE:-checkpoints/medgemma4b_text}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8001}"
export DX_VARIANT=deephedge
COMMON="--base $BASE --env tasks.diagnostics.env_rv --n-cases 300 --max-turns 12 \
        --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --save-steps 20 --max-steps 60 \
        --server-host $SERVER_HOST --server-port $SERVER_PORT"
ALPHA_GRID=(0.05 0.10 0.20)     # small, conservative
VAL_SEED=202                    # validation (alpha + checkpoint selection)
TEST_SEED=100                   # final held-out TEST — do NOT select on this
OUT=checkpoints/procreward
LOG=results/procreward_pilot

if [[ "$CONFIRM" != "--confirm" ]]; then
  cat <<EOF
[DRY RUN] This is the prepared 3-arm plan. Re-run with --confirm to execute (needs the vLLM server up:
  bash framework/infra/serve_b.sh   # on the rollout node, port $SERVER_PORT).

Planned steps:
  0. Prereq: Claude-only smoke shows a measurable, stable effect (scripts/run_procreward_smoke.sh).
  1. ALPHA SWEEP (validation-only): for alpha in ${ALPHA_GRID[*]}, short Claude-GRPO (seed 1, 30 steps),
     probe on VALIDATION seed $VAL_SEED, pick alpha maximizing high-risk value w/ no unsafe-high increase.
  2. THREE ARMS at the selected alpha, identical config:
       plain GRPO (seed 1) | info-gain GRPO (seed 1) | Claude process-reward (seeds 1,2,3)
  3. EVAL every checkpoint on TEST seed $TEST_SEED with support_probe (k=12, temp 1.0) AND greedy pass@1
     (k=1, temp 0), then framework.procreward.eval_metrics + framework.report (guarded verdict).
  4. Do NOT run the combined info-gain+Claude arm yet (gated on a positive Claude-only result).
EOF
  exit 0
fi

mkdir -p "$OUT" "$LOG"
echo "[arms] alpha sweep on validation seed $VAL_SEED ..."
for A in "${ALPHA_GRID[@]}"; do
  "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.rl.grpo_disagg $COMMON --seed 1 --infogain-coef 0.0 \
      --claude-alpha "$A" --claude-cache "$LOG/grpo_judge_cache.jsonl" \
      --claude-log "$LOG/sweep_a${A}_rewardlog.jsonl" --max-steps 30 --out "$OUT/sweep_a${A}"
  "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
      --adapter "$OUT/sweep_a${A}/final" --tag "sweep_a${A}" --k 12 --n-cases 60 \
      --conditions hidden --max-turns 12 --seed "$VAL_SEED" --out "$LOG/val_a${A}.json"
  "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.procreward.eval_metrics --records "$LOG/val_a${A}.records.jsonl"
done
echo "[arms] -> pick ALPHA by validation high-risk value (no unsafe-high increase), set ALPHA below, re-run part 2."

ALPHA="${ALPHA:-0.10}"   # <-- set from the sweep before running the arms
run_arm () { # name  extra-flags  seed
  "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.rl.grpo_disagg $COMMON --seed "$3" $2 --out "$OUT/$1_s$3"
  for SEED in "$TEST_SEED"; do
    "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
        --adapter "$OUT/$1_s$3/final" --tag "$1_s$3" --k 12 --n-cases 60 --conditions hidden \
        --max-turns 12 --seed "$SEED" --out "$LOG/test_${1}_s$3.json"
    "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.gates.support_probe --env tasks.diagnostics.env_rv --model "$BASE" \
        --adapter "$OUT/$1_s$3/final" --tag "${1}_s$3_greedy" --k 1 --n-cases 60 --conditions hidden \
        --max-turns 12 --seed "$SEED" --out "$LOG/test_${1}_s$3_greedy.json"
    "${PY:-$HOME/spark1/.venv/bin/python}" -m framework.procreward.eval_metrics --records "$LOG/test_${1}_s$3.records.jsonl"
  done
}
echo "[arms] three arms at alpha=$ALPHA (test seed $TEST_SEED) ..."
run_arm plain    "--infogain-coef 0.0" 1
run_arm infogain "--infogain-coef 0.2" 1
for S in 1 2 3; do
  run_arm claude "--infogain-coef 0.0 --claude-alpha $ALPHA --claude-cache $LOG/grpo_judge_cache.jsonl \
                  --claude-log $LOG/claude_s${S}_rewardlog.jsonl" "$S"
done
echo "[arms] done. Compare with framework.report + eval_metrics; keep support@12 as a diagnostic only."
