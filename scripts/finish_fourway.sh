#!/usr/bin/env bash
set -uo pipefail
LOG=results/procreward_pilot/sweep
echo "waiting for a0.00 training DONE..."
while ! grep -qa 'DONE steps=' "$LOG/a0.00_train.log" 2>/dev/null; do
  pgrep -f grpo_disagg >/dev/null || { grep -qa 'DONE steps=' "$LOG/a0.00_train.log" 2>/dev/null || echo "grpo gone (no DONE)"; break; }
  sleep 60
done
echo "a0.00: $(grep -a 'DONE steps=' "$LOG/a0.00_train.log" | tail -1 || echo 'NO DONE')"
for i in $(seq 1 30); do pkill -9 -f EngineCore 2>/dev/null; pkill -9 -f vllm 2>/dev/null; \
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l)" = "0" ] && break; sleep 6; done
rm -f /dev/shm/nccl* 2>/dev/null
DX_VARIANT=deephedge PYTHONPATH="$PWD" ~/spark1/.venv/bin/python scripts/ab_fourway.py > results/procreward_pilot/fourway_console.log 2>&1
echo "===== FOURWAY DONE -> results/procreward_pilot/fourway/fourway_report.md ====="
