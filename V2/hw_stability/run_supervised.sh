#!/bin/bash
# Run one workload on Spark 1 under supervision.
#   V2/hw_stability/run_supervised.sh <run_name> <timeout_s> <gpu_max> -- <workload command...>
# - Spark 1 telemetry (1 Hz) is written locally AND streamed to Spark 2, so it survives a power loss here.
# - Spark 2 records its own telemetry for the same window.
# - The workload runs in its own session; watchdog.py (a separate process) kills that process group on a trip.
# - `timeout` bounds the workload's wall clock independently of both.
# Results: results/v2_hw/<run_name>/ here and ~/v2_hw/<run_name>/ on Spark 2.
set -u
NAME=$1; TMO=$2; GMAX=$3; shift 4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$REPO/results/v2_hw/$NAME; mkdir -p "$OUT"
PY=~/spark1/.venv/bin/python
S2=192.168.100.2
ssh -o BatchMode=yes $S2 "mkdir -p ~/v2_hw/$NAME"
scp -q "$REPO/V2/hw_stability/telemetry.py" $S2:~/v2_hw/telemetry.py

# Spark 2 local telemetry, bounded
ssh -o BatchMode=yes $S2 "nohup timeout $((TMO+300)) python3 ~/v2_hw/telemetry.py > ~/v2_hw/$NAME/spark2.csv 2>&1 &"
# Spark 1 telemetry: local file + live copy on Spark 2
setsid bash -c "echo \$\$ > '$OUT/telemetry.pgid'; timeout $((TMO+300)) python3 '$REPO/V2/hw_stability/telemetry.py' | \
    tee '$OUT/spark1.csv' | ssh -o BatchMode=yes $S2 'cat > ~/v2_hw/$NAME/spark1_mirror.csv'" &
sleep ${BASELINE:-30}   # idle baseline in the log

PGF=$OUT/workload.pgid
touch "$PGF.running"
setsid bash -c "echo \$\$ > '$PGF'; exec timeout --kill-after=10 $TMO $*" > "$OUT/workload.log" 2>&1 &
WL=$!
sleep 1
$PY "$REPO/V2/hw_stability/watchdog.py" --pgid-file "$PGF" --log "$OUT/watchdog.log" --gpu-max "$GMAX" \
    --max-seconds $((TMO+30)) &
WD=$!
wait $WL; RC=$?
rm -f "$PGF.running"
echo "$(date -u +%FT%T) workload exit $RC" >> "$OUT/watchdog.log"
wait $WD
sleep ${COOLDOWN:-120}  # cool-down in the log
kill -- -$(cat "$OUT/telemetry.pgid") 2>/dev/null
ssh -o BatchMode=yes $S2 "pkill -u \$USER -f 'v2_hw/telemetry.py' ; true"
echo "RC=$RC"; cat "$OUT/watchdog.log"
[ -f "$OUT/watchdog.log.TRIPPED" ] && echo "TRIPPED: $(cat "$OUT/watchdog.log.TRIPPED")"
exit $RC
