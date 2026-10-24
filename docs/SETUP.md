# Studio setup

## ComfyUI

We run stock ComfyUI **0.36.0** (a clone of upstream at commit `3dd559d`, no local changes). MiniMax H3 and
Qwen-Image-2.1 support are both built in (`comfy_extras/nodes_minimax_h3.py`, the `qwen_image` text encoder type,
`TextEncodeQwenImage21`). The older ComfyUI 0.30 we first used fails on the Qwen-Image-2.1 VAE (`WanVAE size mismatch`).

ComfyUI 0.36 pins newer helper packages than our image had. We install them into a side folder and pass it with
`PYTHONPATH`, so the image stays untouched:

```bash
pip install --target ~/studio/pydeps \
  comfy-kitchen==0.2.35 comfy-aimdo==0.5.5 comfyui-frontend-package==1.53.6 \
  comfyui-workflow-templates==0.11.66 comfyui-embedded-docs==0.5.12
```

Layout on each Spark:

```
~/studio/ComfyUI        ComfyUI 0.36.0 (its models/ is a symlink to ~/h3/ComfyUI/models)
~/studio/pydeps         the packages above
~/h3/ComfyUI/models     diffusion_models/ text_encoders/ vae/ loras/
~/h3/h3_guard.sh        the guard (scripts/h3_guard.sh)
```

## The container image

Any arm64 image with a PyTorch build that supports GB10 (CUDA, sm_121) and ComfyUI's `requirements.txt` works. Ours,
`comfy-h3:gb10`, is a vLLM runtime image for the Spark (a micromamba environment at `/opt/env`) with ComfyUI's
requirements added. Two things to know about it:

- its default entrypoint is `sleep 7200`, so `run_comfy_studio.sh` overrides `--entrypoint /opt/env/bin/python3`;
- set `STUDIO_IMAGE` to your own image name if you build a different one.

## Model files

`scripts/get_models.sh` downloads these (about 67 GB) with resumable HTTPS:

| file | from | size |
|---|---|---|
| `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | |
| `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors` | Comfy-Org/Qwen-Image-2.1 | |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | Comfy-Org/Qwen-Image-2.1 | |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | Comfy-Org/Qwen-Image-2.1 | Qwen total ~24.6 GB |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | 21 GB |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Comfy-Org/MiniMax-H3 | 15.7 GB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | Comfy-Org/MiniMax-H3 | 5.2 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 | 0.6 GB |

Optional LoRAs (style, character swap, turbo schedules) go in `loras/`. A job that names a LoRA fails on a Spark that does
not have the file, so keep the four Sparks identical (`scripts/push_studio.sh`).

On our Sparks the Hugging Face "xet" transport crawled at about 1 MB/s; plain HTTPS (or `HF_HUB_DISABLE_XET=1` with the
Python client) ran at about 110 MB/s. Download once, then copy to the other Sparks over the RoCE fabric.

## Bring-up order after a reboot

1. DeepSeek TP4 (the recipe's `serve.sh start`), wait for it to answer.
2. On each Spark: `docker start comfy-studio` (or `scripts/run_comfy_studio.sh` the first time), then `scripts/start_guard.sh`.
3. `scripts/studio_status.py`.

The studio containers use `--restart no` on purpose: they must never come up before the rank and take its memory.
