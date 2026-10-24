# Credits

This setup is glue around other people's work. Our part is the memory budget, the launch flags, the guard, the
scheduling rule and the measurements. Everything else is theirs.

## Models

- **DeepSeek-AI**: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and its 1017 update (MIT).
- **LibertAIDAI**: [DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) (MIT), the weights our TP4 lane serves.
- **Qwen team (Alibaba)**: Qwen-Image-2.1; **Comfy-Org** for the ComfyUI packaging, [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). See the model card for the license.
- **MiniMax**: MiniMax H3; **Comfy-Org** for the ComfyUI packaging, [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). See the model card for the license.

## Software

- **TensorFold** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) through **bertholomus**'s DeepSeek-V4.1 fork and TP4 recipe ([bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), [deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)): the engine that serves DeepSeek here. The full credits for the DeepSeek lane (weights, engine patches, the people whose fixes it uses) are in the [DeepSeek recipe's CREDITS](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks/blob/main/CREDITS.md).
- **ComfyUI** ([comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), GPL-3.0), version 0.36.0, unmodified. This repository contains no ComfyUI code; our scripts talk to its HTTP API.
- **NVIDIA**: DGX Spark / GB10 and the PyTorch container stack.

## Ours

The scripts, examples, benchmark and docs in this repository (MIT, [LICENSE](LICENSE)). The ComfyUI graphs in
`examples/graphs.py` are our own, adapted from our [2Wild-Image-Video-Gen-Studio](https://github.com/tonyd2wild/2Wild-Image-Video-Gen-Studio).

If we missed you or described your work wrong, please open an issue.
