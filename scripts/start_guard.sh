#!/bin/bash
# start_guard.sh [container]: start h3_guard.sh for the studio container unless one already watches it, detached from
# this shell (setsid + nohup, so it survives the SSH session ending). Prints the guard PID and the ComfyUI status code.
N=${1:-comfy-studio}
G="$(cd "$(dirname "$0")" && pwd)/h3_guard.sh"
P="^/bin/bash $G $N"
if ! pgrep -f "$P" >/dev/null; then
  setsid nohup /bin/bash "$G" "$N" >/dev/null 2>&1 < /dev/null &
fi
sleep 4
echo "$(hostname) guard: $(pgrep -f "$P" | tr '\n' ' ') comfy=$(curl -s -m 10 -o /dev/null -w %{http_code} http://127.0.0.1:${STUDIO_PORT:-8188}/system_stats)"
