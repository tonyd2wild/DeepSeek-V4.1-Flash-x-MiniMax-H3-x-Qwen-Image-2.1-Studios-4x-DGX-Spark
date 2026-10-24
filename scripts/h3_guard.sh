#!/bin/bash
# h3_guard.sh [container] [floor_mb]
# Memory guard for a ComfyUI container that shares a DGX Spark with a TensorFold rank.
# GB10 has one memory pool for CPU and GPU. If a render pushes MemAvailable under the floor, this kills the
# ComfyUI container (never the DeepSeek rank) and exits. Logs: ~/h3/logs/guard_mem.log (1 line/s) and guard.log.
N=${1:-comfy-studio}
FLOOR_MB=${2:-4096}
mkdir -p ~/h3/logs
sleep 3
while [ -n "$(docker ps -q -f name=^${N}$)" ]; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  echo "$(date +%T) avail_mb=$a" >> ~/h3/logs/guard_mem.log
  if [ "$a" -lt "$FLOOR_MB" ]; then
    docker kill "$N"
    echo "$(date) KILLED $N avail_mb=$a" >> ~/h3/logs/guard.log
    break
  fi
  sleep 1
done
echo "$(date) guard exit" >> ~/h3/logs/guard.log
