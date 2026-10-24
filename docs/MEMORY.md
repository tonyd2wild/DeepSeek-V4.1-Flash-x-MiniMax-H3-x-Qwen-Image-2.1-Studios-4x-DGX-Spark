# The memory budget

## One pool, two users

A DGX Spark (GB10) has 128 GB of LPDDR5X that the CPU and the GPU share. There is no separate VRAM: when PyTorch
allocates "GPU memory", it comes out of the same pool the operating system calls RAM, and `MemAvailable` in
`/proc/meminfo` goes down by the same amount. Linux reports about 121.7 GiB total.

On our Sparks two processes share that pool:

1. **The DeepSeek rank.** TensorFold allocates the rank's share of the weights, the KV pool (2,097,152 tokens across
   the four ranks, 1,048,576 per request, 16 streams) and its working buffers once, at boot. `nvidia-smi` shows about
   73.5 GB for the rank process. It does not grow while serving, which is what makes everything else plannable.
2. **The studio.** ComfyUI 0.36 with Qwen-Image-2.1 and MiniMax H3. It loads weights when a job needs them and keeps them
   until told to free them.

While DeepSeek serves, MemAvailable sits around 35-42 GB per Spark (it differs a little per node). That is the studio's
whole budget.

## What a studio job takes

| job | resident while rendering | lowest MemAvailable seen | notes |
|---|---|---|---|
| Qwen-Image-2.1 turbo, 1328x1328 | ~19 GB | 20.6 GB | INT8 diffusion model + 8B INT8 text encoder + VAE |
| MiniMax H3, 2.3 s clip with audio | ~34 GB | 7.4-8.0 GB | weights load in stages (text encoder ~15 GB, then the 21 GB diffusion model, then the VAEs); activations are small at this length |

After an H3 job the weights stay resident (about 8 GB left). `POST /free` on ComfyUI brings MemAvailable back to about
36 GB without restarting anything.

## Pinned memory

ComfyUI's default on Linux is to page-lock ("pin") host memory for faster transfers, up to roughly 90% of system RAM.
On a discrete GPU that costs RAM; on GB10 the "host RAM" is the same pool the DeepSeek rank is using, so pinning starves
it. `--disable-pinned-memory` turns that off. On an RTX 3090 box the same flag took a 15 s H3 render from an OOM kill
(29.9 of 32 GB RAM) to finishing in 7.5 GB ([MiniMax-H3-Local](https://github.com/tonyd2wild/MiniMax-H3-Local)).

The other ComfyUI flags:

- `--reserve-vram 48`: ComfyUI's loader treats 48 GB of the reported device memory as off limits when deciding what to
  keep loaded.
- `--fp16-intermediates`: keeps H3's intermediate tensors in fp16.

## The guard and the kernel

Two layers make sure that if memory does run out, the studio loses, not the LLM:

- `scripts/h3_guard.sh` reads MemAvailable every second and `docker kill`s the studio container under 4 GB.
- The container runs with `--oom-score-adj 800`, so if the kernel's OOM killer ever fires first, ComfyUI is its target.

The guard has fired once so far (2026-10-21: a studio job took MemAvailable down to 4.05 GB and the guard killed the studio, as designed). The kernel logs on the four Sparks show no OOM kills.

## What fits on one Spark

```
~121.7 GiB total
 - ~74   DeepSeek rank (fixed)
 - ~10   OS, Docker, the rest
 = ~35-42 for the studio
     one picture   ~19 -> ~15-20 spare
     one video     ~34 -> ~4-8 above the guard floor
     two jobs      do not fit
```

That is the reason for the scheduling rule "one job per Spark", and why four slots means one per Spark.

## Things we have not measured yet

- Longer H3 clips (5-15 s) next to the rank. Activations grow with length; the 2.3 s numbers above do not cover them.
- DeepSeek's throughput while studios render on this TensorFold lane.
