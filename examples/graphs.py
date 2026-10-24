"""ComfyUI API graphs for the studio (ComfyUI 0.36): Qwen-Image-2.1 (normal + turbo) and MiniMax H3 text-to-video.
File names match docs/SETUP.md. Stdlib only; import from the other scripts."""

QWEN_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_TURBO_UNET = "qwen_image_2.1_turbo_int8_convrot.safetensors"
QWEN_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"

H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
H3_VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
H3_LENGTHS = [56, 124, 192, 243, 311, 362]   # 17n+5 frames at 24 fps: 2.3 s ... 15.1 s


def qwen(prompt, seed, width=1328, height=1328, turbo=True, steps=20, cfg=2.5, prefix="studio/IMG"):
    """Text-to-image. turbo=True: the turbo checkpoint with its fixed 8-sigma schedule (cfg 1)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_TURBO_UNET if turbo else QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }
    if turbo:
        g["11"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["4", 0]}}
        g["12"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN_TURBO_SIGMAS}}
        g["13"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        g["14"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        g["8"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["13", 0],
                                                                     "sigmas": ["12", 0], "latent_image": ["6", 0]}}
    else:
        g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}}
        g["7"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}}
        g["8"] = {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                                                        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                                                        "scheduler": "simple", "denoise": 1.0}}
    return g


def h3(prompt, seed, width=960, height=544, length=56, steps=20, prefix="studio/VID"):
    """MiniMax H3 text-to-video with audio: fl2va checkpoint + NVFP4 text encoder, res_multistep / simple, no LoRA."""
    return {
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_VIDEO}},
        "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_AUDIO}},
        "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": prompt,
                                                                   "width": width, "height": height, "length": length}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": ["104", 0]}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler", "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                                                                  "sigmas": ["9", 0], "latent_image": ["104", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24.0}},
        "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
