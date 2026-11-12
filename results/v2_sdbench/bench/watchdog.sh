#!/bin/bash
# Kill eval clients if Spark 2's GPU reaches 85 C or stops answering; log temps every 30 s.
LOG=$1
while true; do
  t=$(ssh -o ConnectTimeout=5 192.168.100.2 'nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader' 2>/dev/null)
  echo "$(date -u +%H:%M:%S) ${t:-UNREACHABLE}" >> $LOG
  temp=${t%%,*}
  if [ -z "$t" ] || [ "$temp" -ge 85 ]; then
    echo "$(date -u +%H:%M:%S) WATCHDOG KILL temp=${temp:-none}" >> $LOG
    pkill -f "tasks.v2_sdbench.run_eval"; echo "WATCHDOG KILL temp=${temp:-unreachable}"
  fi
  sleep 30
done
