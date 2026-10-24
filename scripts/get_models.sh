#!/bin/bash
# get_models.sh [models_dir]: download the studio's model files (about 67 GB) into a ComfyUI models folder.
#   Qwen-Image-2.1 from Comfy-Org/Qwen-Image-2.1 (INT8 normal + turbo, the 8B text encoder, the VAE)   ~24.6 GB
#   MiniMax H3     from Comfy-Org/MiniMax-H3      (fl2va INT8, the NVFP4 32B text encoder, video + audio VAE) ~42 GB
# HF_HUB_DISABLE_XET-style plain HTTPS with resume: on our Sparks the xet transport crawled at ~1 MB/s.
set -u
M=${1:-$HOME/h3/ComfyUI/models}
mkdir -p "$M/diffusion_models" "$M/text_encoders" "$M/vae" "$M/loras"
dl() {  # repo path
  echo "== $2"
  curl -sL --retry 5 -C - -o "$M/$2" "https://huggingface.co/$1/resolve/main/$2" && ls -la "$M/$2" | awk '{print $5, $9}'
}
Q=Comfy-Org/Qwen-Image-2.1
dl $Q diffusion_models/qwen_image_2.1_int8_convrot.safetensors
dl $Q diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors
dl $Q text_encoders/qwen3vl_8b_int8_convrot.safetensors
dl $Q vae/qwen_image_2.1_vae_bf16.safetensors
H=Comfy-Org/MiniMax-H3
dl $H diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
dl $H text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
dl $H vae/minimax_h3_video_vae_fp16.safetensors
dl $H vae/minimax_h3_audio_vae_fp32.safetensors
echo "done. Copy once, then rsync to the other Sparks over the fabric (scripts/push_studio.sh)."
