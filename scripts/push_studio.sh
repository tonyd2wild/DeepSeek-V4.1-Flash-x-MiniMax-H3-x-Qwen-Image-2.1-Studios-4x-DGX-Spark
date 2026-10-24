#!/bin/bash
# push_studio.sh <user@fabric-ip> <remote_home>: copy the studio (ComfyUI, pydeps, model files, guard) from this Spark
# to another one over the RoCE fabric (we saw ~300 MB/s). Run on the Spark that already has everything.
set -u
T=$1; RH=$2
M=$HOME/h3/ComfyUI/models
RS="rsync -a --partial --info=progress2"
ssh "$T" "mkdir -p $RH/h3/ComfyUI/models/diffusion_models $RH/h3/ComfyUI/models/text_encoders $RH/h3/ComfyUI/models/vae $RH/h3/ComfyUI/models/loras $RH/studio $RH/h3/logs"
for d in diffusion_models text_encoders vae loras; do $RS "$M/$d/" "$T:$RH/h3/ComfyUI/models/$d/"; done
rsync -a --exclude=/ComfyUI/models --exclude=/ComfyUI/output --exclude=/ComfyUI/temp --exclude=/ComfyUI/user --exclude=/ComfyUI/input \
  "$HOME/studio/ComfyUI" "$HOME/studio/pydeps" "$T:$RH/studio/"
ssh "$T" "cd $RH/studio/ComfyUI && ln -sfn $RH/h3/ComfyUI/models models && mkdir -p input output temp user"
echo "pushed to $T"
