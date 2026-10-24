#!/bin/bash
# run_comfy_studio.sh: start the studio (ComfyUI 0.36 with Qwen-Image-2.1 + MiniMax H3) on ONE Spark, next to its
# running TensorFold DeepSeek rank. Run it on each Spark. Then start the guard: scripts/start_guard.sh
#
#   STUDIO_IMAGE   any arm64 image with a GB10 (sm_121) CUDA PyTorch at /opt/env/bin/python3 (see docs/SETUP.md)
#   STUDIO_HOME    holds h3/ComfyUI/models (the model files) and studio/ComfyUI + studio/pydeps
#   COMFY_DIR      the ComfyUI 0.36 folder (default $STUDIO_HOME/studio/ComfyUI)
#   STUDIO_PORT    8188
#
# The flags are the whole trick (docs/MEMORY.md):
#   --disable-pinned-memory   ComfyUI otherwise page-locks up to ~90% of system RAM; on GB10 that RAM IS the GPU memory
#                             the DeepSeek rank lives in
#   --reserve-vram 48         ComfyUI plans as if 48 GB of the pool is not there (it is the rank's)
#   --fp16-intermediates      smaller activations for the H3 decode
#   --oom-score-adj 800       if the kernel must kill something, it kills ComfyUI, not the rank
#   --restart no              the studio never comes back on its own after a reboot (bring DeepSeek up first)
set -eu
H=${STUDIO_HOME:-$HOME}
IMAGE=${STUDIO_IMAGE:-comfy-h3:gb10}
PORT=${STUDIO_PORT:-8188}
COMFY_DIR=${COMFY_DIR:-$H/studio/ComfyUI}
MODELS_EXTRA=${MODELS_EXTRA:-/var/tmp/models}     # mounted too, for model folders that are symlinked there
if docker ps -a --format '{{.Names}}' | grep -qx comfy-studio; then
  echo "comfy-studio already exists: docker start comfy-studio (or docker rm -f comfy-studio to recreate)" >&2; exit 1
fi
docker run -d --name comfy-studio --network host --gpus all --oom-score-adj 800 --restart no \
  -e PYTHONPATH="$H/studio/pydeps" \
  -v "$H/h3:$H/h3" -v "$H/studio:$H/studio" -v "$MODELS_EXTRA:$MODELS_EXTRA" \
  -w "$COMFY_DIR" --entrypoint /opt/env/bin/python3 "$IMAGE" \
  main.py --listen 0.0.0.0 --port "$PORT" --disable-pinned-memory --fp16-intermediates --reserve-vram 48
for i in $(seq 1 60); do
  curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/system_stats" && { echo "comfy-studio up on :$PORT after ${i}s"; exit 0; }
  sleep 1
done
echo "comfy-studio did not answer on :$PORT, see: docker logs comfy-studio" >&2; exit 1
