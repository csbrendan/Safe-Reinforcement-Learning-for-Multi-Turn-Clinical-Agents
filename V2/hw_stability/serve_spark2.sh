#!/bin/bash
# Start the V2 SD-Bench inference servers on Spark 2 under the independent watchdog (run FROM Spark 1).
#   V2/hw_stability/serve_spark2.sh start [max_hours]   # Qwen3-8B :8101 (0.45) + MedGemma :8100 (0.20), one pgid
#   V2/hw_stability/serve_spark2.sh stop
# Same args as the 2026-11-11/12 runs, so the gatekeeper/judge caches stay valid. MedGemma is served from
# medgemma4b_text_v2stop (hard links to the same weights; generation_config adds <end_of_turn>=106 as EOS) because a
# grammar-constrained decode can only end on an EOS id, and stop_token_ids are not EOS ids. Logs and telemetry: ~/v2_hw/serve/
# on Spark 2. Thermal governor: the server group is paused (SIGSTOP) at 80 C and resumed at 74 C (GPU or ACPI;
# the GPU keeps running already-queued vLLM steps for 3-4 s after SIGSTOP, hence the 8 C margin);
# the watchdog kills the whole group at >= 85 C (the stop rule for these tests; firmware critical is 104.8 C), on throttle bits, kernel errors, or the
# wall-clock cap.
set -u
S2=192.168.100.2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "${1:-}" in
start)
  HRS=${2:-6}
  scp -q "$REPO/V2/hw_stability/watchdog.py" "$REPO/V2/hw_stability/telemetry.py" $S2:~/v2_hw/
  ssh -o BatchMode=yes $S2 "bash -s" <<EOF
set -u
D=~/v2_hw/serve; mkdir -p \$D; cd \$D; rm -f srv.pgid srv.pgid.running watchdog.log.TRIPPED
V=~/spark1/.venv/bin/vllm; M=~/v2_bench/models
touch srv.pgid.running
setsid bash -c "echo \\\$\\\$ > srv.pgid
  \$V serve \$M/qwen3-8b --host 0.0.0.0 --port 8101 --dtype bfloat16 --seed 1 --max-model-len 32768 \
     --served-model-name local-aux --gpu-memory-utilization 0.45 > qwen.log 2>&1 &
  until curl -sf localhost:8101/v1/models >/dev/null; do sleep 5; done
  \$V serve \$M/${MEDGEMMA_DIR:-medgemma4b_text_v2stop} --host 0.0.0.0 --port 8100 --dtype bfloat16 --seed 1 --max-model-len 16384 \
     --served-model-name base --gpu-memory-utilization 0.2 > medgemma.log 2>&1 &
  wait" > /dev/null 2>&1 < /dev/null &
sleep 2
nohup timeout $((HRS*3600+600)) python3 ~/v2_hw/telemetry.py > telemetry.csv 2>/dev/null < /dev/null &
nohup ~/spark1/.venv/bin/python ~/v2_hw/watchdog.py --pgid-file srv.pgid --log watchdog.log --gpu-max 85 --pause-at 80 --resume-at 74 \
   --acpi-max 85 --gpu-warn 78 --max-seconds $((HRS*3600)) > /dev/null 2>&1 < /dev/null &
echo started pgid \$(cat srv.pgid)
EOF
  ;;
stop)
  ssh -o BatchMode=yes $S2 'cd ~/v2_hw/serve; rm -f srv.pgid.running; kill -- -$(cat srv.pgid) 2>/dev/null; sleep 5;
     kill -9 -- -$(cat srv.pgid) 2>/dev/null; for p in $(pgrep -f "v2_hw/telemetry.py"); do [ "$p" != "$$" ] && kill $p; done;
     nvidia-smi --query-compute-apps=pid --format=csv,noheader'
  ;;
*) echo "usage: $0 start [hours] | stop"; exit 2;;
esac
