#!/usr/bin/env bash
# =============================================================================
# Claude process-reward GRPO — reward-integration LIVE SMOKE (tiny, end-to-end).
# Confirms the wired reward path trains a few steps WITHOUT NaN and logs the
# raw/normalized/analytic/total components. NOT the full run. Guarded: needs
# --confirm. Requires the vLLM rollout server up first:
#     bash framework/infra/serve_b.sh        # port 8001 by default
#
# The CPU math of the integration is already verified deterministically:
#     PYTHONPATH=$PWD python -m framework.procreward.smoke_reward   # 15/15 PASS
# This script exercises the SAME path end-to-end through the trainer + judge.
# =============================================================================
set -euo pipefail
CONFIRM="${1:-}"
BASE="${BASE:-checkpoints/medgemma4b_text}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8001}"
export DX_VARIANT=deephedge
LOG=results/procreward_pilot

if [[ "$CONFIRM" != "--confirm" ]]; then
  echo "[DRY RUN] live reward-integration smoke. Re-run with --confirm (vLLM server must be up on :$SERVER_PORT)."
  echo "  It runs: 3 GRPO steps, 8 cases, alpha=0.1, and checks [claude-reward] logs + no NaN."
  exit 0
fi

source ~/.secrets/anthropic.env
mkdir -p "$LOG"
"${PY:-$HOME/spark1/.venv/bin/python}" -m framework.rl.grpo_disagg \
  --base "$BASE" --env tasks.diagnostics.env_rv \
  --server-host "$SERVER_HOST" --server-port "$SERVER_PORT" \
  --n-cases 8 --max-turns 12 --max-steps 3 --save-steps 3 \
  --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --seed 1 \
  --infogain-coef 0.0 --claude-alpha 0.1 --claude-norm zscore_clip \
  --claude-cache "$LOG/grpo_judge_cache.jsonl" --claude-log "$LOG/smoke_rewardlog.jsonl" \
  --out checkpoints/procreward_smoke 2>&1 | tee "$LOG/smoke_run.log"

echo "=== smoke checks ==="
grep -q "\[claude-reward\]" "$LOG/smoke_run.log" && echo "PASS: claude-reward component logged" || echo "FAIL: no claude-reward log line"
grep -qi "nan" "$LOG/smoke_run.log" && echo "WARN: 'nan' appears in log — inspect" || echo "PASS: no NaN in log"
test -s "$LOG/smoke_rewardlog.jsonl" && echo "PASS: per-rollout reward log written ($(wc -l < "$LOG/smoke_rewardlog.jsonl") rows)" || echo "FAIL: empty reward log"
