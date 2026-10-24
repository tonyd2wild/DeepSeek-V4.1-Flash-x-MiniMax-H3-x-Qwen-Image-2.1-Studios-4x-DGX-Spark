# DeepSeek V4.1 Flash x MiniMax H3 x Qwen Image 2.1 on 4 DGX Sparks

![1 LLM + 4 Studios: DeepSeek V4.1 Flash across four DGX Sparks, each also running a Qwen Image 2.1 / MiniMax H3 studio](docs/banner.jpg)

**One 1M-context LLM and four image/video studios on the same four NVIDIA DGX Sparks, at the same time.**

DeepSeek-V4.1-Flash runs tensor-parallel across all four Sparks (TensorFold TP4, up to 1,048,576 tokens of context per
request, a 2,097,152-token KV pool, 16 streams; our agents run at 300K per request day to day, 1M is the ceiling when needed). Next to each DeepSeek rank, every Spark also runs its own ComfyUI "studio" that holds
**Qwen-Image-2.1** (pictures) and **MiniMax H3** (video with audio). That gives four independent media slots: four pictures
at once, four videos at once, or any mix, while the LLM keeps serving.

The trick is not a new kernel. It is memory discipline on GB10's single CPU+GPU memory pool: a fixed budget per Spark,
ComfyUI flags that stop it from grabbing that pool, a guard that kills the studio (never the LLM) if memory runs low, and
one rule for the scheduler: **one job per Spark**.

```
            4 x DGX Spark (GB10, 128 GB unified memory each), one 200G RoCE switch
 ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
 │ Spark 1 (rank 0) │ Spark 2 (rank 1) │ Spark 3 (rank 2) │ Spark 4 (rank 3) │
 │ DeepSeek TP4     │ DeepSeek TP4     │ DeepSeek TP4     │ DeepSeek TP4     │   <- one model, :8000 on rank 0
 │  ~74 GB          │  ~74 GB          │  ~74 GB          │  ~74 GB          │
 ├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
 │ studio :8188     │ studio :8188     │ studio :8188     │ studio :8188     │   <- four independent slots
 │ Qwen 2.1 | H3    │ Qwen 2.1 | H3    │ Qwen 2.1 | H3    │ Qwen 2.1 | H3    │
 │ guard @ 4 GB     │ guard @ 4 GB     │ guard @ 4 GB     │ guard @ 4 GB     │
 └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

## Status

| what | status |
|---|---|
| DeepSeek V4.1 Flash TP4 serving on 4 Sparks (1M context, 2,097,152 KV pool, 16 streams) | **in daily use** since 2026-10-17. Recipe, numbers and evals: [DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks) |
| One studio per Spark (ComfyUI 0.36, Qwen-Image-2.1 + MiniMax H3) next to the rank | **running** since 2026-10-20; our studio web app and our agents' image/video tools send jobs to it |
| A picture or a video on any one Spark while DeepSeek serves | **measured**, see [Numbers](#numbers) |
| Pictures on two or more Sparks at the same time | **measured** (2 at once on 2026-10-23, 2 at once on 2026-10-24) |
| All four Sparks rendering at the same time (4 pictures or 4 videos) | **not yet proven.** Our first attempt (2026-10-24) collided with other users' jobs: 2 Sparks finished, 2 studio processes went into sticky CUDA errors and needed a restart. Clean re-run pending |
| DeepSeek's speed while studios render | **not measured cleanly yet** (only an older lane's curve, below) |
| The same setup on 2 Sparks with a smaller quant | **not built**, see [docs/TWO-SPARKS.md](docs/TWO-SPARKS.md) |

We will update this table as runs land. Nothing above "measured" is a claim.

## The memory budget, per Spark

GB10 has one 128 GB pool shared by the CPU and the GPU (Linux reports about 121.7 GiB). Everything below comes out of it.

| | GB | notes |
|---|---|---|
| DeepSeek TP4 rank (weights share, KV pool share, engine) | ~74 | `nvidia-smi` shows ~73.5 GB for the rank process; it is allocated once at boot and does not grow |
| OS, Docker, everything else | ~10 | |
| **Left for the studio** (MemAvailable while DeepSeek serves) | **~35-42** | varies by node; this is the number every check below uses |
| Qwen-Image-2.1 turbo, loaded and rendering 1328x1328 | ~19 resident | lowest MemAvailable seen during a render: 20.6 GB |
| MiniMax H3 2.3 s clip (fl2va INT8 + NVFP4 text encoder) | ~34 resident | lowest MemAvailable seen: 7.4-8.0 GB |
| Guard floor | 4 | below this the guard kills the studio container |

So a Spark holds **one** studio job at a time: a picture leaves ~15-20 GB spare, a video leaves ~4-8 GB above the guard.
Two jobs on one Spark do not fit. That is why the slots are per Spark, and why "four at once" means one per Spark.

Full explanation, including why pinned memory matters on unified memory: [docs/MEMORY.md](docs/MEMORY.md).

## The settings that make it fit

**DeepSeek (TensorFold TP4)**, from the recipe's `config.env`:

```bash
CONTEXT=1048576          # per-request ceiling (our agents send up to 300K; 1M is there when needed)
WINDOW=2097152           # KV pool, shared by all streams
PARALLEL=16              # streams
FOLLOW_FLAGS="--round-rows 48 --streams 16 --pool 2097152 --context 1048576"
```

The KV pool is allocated once when the lane boots, so the rank's footprint is fixed. That is what makes the rest of the
Spark predictable. (Launch, engine, weights: the [DeepSeek recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks).)

**The studio (ComfyUI 0.36)**, [`scripts/run_comfy_studio.sh`](scripts/run_comfy_studio.sh):

```bash
docker run -d --name comfy-studio --network host --gpus all --oom-score-adj 800 --restart no ... \
  main.py --listen 0.0.0.0 --port 8188 --disable-pinned-memory --fp16-intermediates --reserve-vram 48
```

| flag | why |
|---|---|
| `--disable-pinned-memory` | ComfyUI page-locks up to ~90% of system RAM by default. On GB10 that RAM **is** the memory the DeepSeek rank lives in. This one flag is the difference between co-existing and the kernel OOM-killer ([write-up on a 3090](https://github.com/tonyd2wild/MiniMax-H3-Local)) |
| `--reserve-vram 48` | ComfyUI plans its weight loading as if 48 GB of the pool were not there |
| `--fp16-intermediates` | smaller activations for H3's decode |
| `--oom-score-adj 800` | if the kernel ever has to kill something, it picks ComfyUI, not the rank |
| `--restart no` | the studio never comes back by itself after a reboot; bring DeepSeek up first, then the studios |

**The guard**, [`scripts/h3_guard.sh`](scripts/h3_guard.sh): checks MemAvailable every second and `docker kill`s the studio
if it drops under 4 GB. Start it detached with [`scripts/start_guard.sh`](scripts/start_guard.sh).

## Scheduling: one job per Spark

Whatever sends jobs (a web app, an agent tool, a script) follows the same rules
([`examples/studio_dispatch.py`](examples/studio_dispatch.py) is a working reference, stdlib only):

1. A Spark gets a job only when its own ComfyUI queue is empty.
2. ...and only when MemAvailable covers it. ComfyUI reports MemAvailable as `system.ram_free` in `/system_stats`; use
   that, not `vram_free`, which reads much lower while weights stream in and is not what the guard watches.

   | job | needs (cold) | needs (same model already loaded) |
   |---|---|---|
   | picture (Qwen-Image-2.1) | 26 GB | 8 GB |
   | video (MiniMax H3, 2.3 s) | 37.5 GB | 6 GB |

   plus the 4 GB guard floor.
3. Switching a Spark between Qwen and H3: `POST /free` first, so only one model family is resident.
4. After a video: `POST /free` (H3 keeps ~34 GB resident otherwise). Idle picture weights: free them after ~10 minutes.

```bash
STUDIOS="http://spark1:8188 http://spark2:8188 http://spark3:8188 http://spark4:8188"
python3 examples/studio_dispatch.py --images 4 --videos 0     # four pictures, one per Spark
python3 examples/studio_dispatch.py --images 0 --videos 4     # four H3 clips, one per Spark
python3 examples/studio_dispatch.py --images 2 --videos 2     # any mix
```

## Quick start

1. **DeepSeek first.** Bring up the TP4 lane from the [DeepSeek recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks) and check it answers.
2. **Models** (one Spark): [`scripts/get_models.sh`](scripts/get_models.sh) pulls Qwen-Image-2.1 and MiniMax H3 from Comfy-Org (~67 GB).
3. **ComfyUI 0.36 + its pinned Python packages** in a GB10 PyTorch image: [docs/SETUP.md](docs/SETUP.md).
4. **Copy to the other Sparks** over the fabric: [`scripts/push_studio.sh`](scripts/push_studio.sh).
5. **On each Spark:** `scripts/run_comfy_studio.sh`, then `scripts/start_guard.sh`.
6. **Check:** `STUDIOS=... DEEPSEEK=http://spark1:8000 python3 scripts/studio_status.py`
7. **Send jobs** with the rules above, e.g. `examples/studio_dispatch.py`.

## Numbers

Measured on our four Sparks with DeepSeek serving real traffic. "Cold" includes loading the model.

| test | Spark(s) | result |
|---|---|---|
| Qwen-Image-2.1 turbo, 1328x1328, 8 steps (2026-10-20 setup test) | each of the 4 | 41 / 43 / 46 / 47 s; lowest MemAvailable 20.6 GB |
| Qwen-Image-2.1 turbo through the studio web app (2026-10-23) | 1 | 18 s |
| Qwen-Image-2.1 turbo, 1328x1328, cold, two Sparks at once (2026-10-24) | 2 | 34.5 s and 35.7 s |
| MiniMax H3 2.3 s clip with audio, 640x352, 20 steps, cold (2026-10-20) | 1 | 123-136 s; lowest MemAvailable 7.4-7.8 GB; DeepSeek `/health` stayed ok |
| MiniMax H3 2.3 s clip, studio defaults, cold (2026-10-23) | 1 | 4 min 31 s |

What a render costs DeepSeek has not been measured cleanly on this lane. On an older vLLM DeepSeek lane (August 2026), one
H3 render on one of the four nodes cost about 34% of aggregate throughput: with tensor parallelism, every step waits for
the slowest rank.

## Gotchas we hit

- **Pinned memory** (above). Without `--disable-pinned-memory`, ComfyUI and the LLM fight over the same pool.
- **Sticky CUDA errors.** In our first four-at-once attempt, two studio processes started failing every job instantly
  (`Allocation on device 0 would exceed allowed memory` on a 4-byte request with 38 GB free; `CUBLAS_STATUS_INTERNAL_ERROR`)
  and stayed that way until the container was restarted. Restart the studio (`docker restart comfy-studio`) and its guard;
  DeepSeek is not affected. Root cause still open.
- **The guard watches MemAvailable**, and so must your scheduler (`ram_free`, not `vram_free`).
- **The old ComfyUI v0.30** in our image fails on the Qwen-Image-2.1 VAE (`WanVAE size mismatch`); 0.36 runs both models.
- **Our image's default entrypoint is `sleep 7200`** (it was built from a vLLM runtime). Always override `--entrypoint`.

## Repository layout

```
scripts/run_comfy_studio.sh   start the studio on one Spark
scripts/h3_guard.sh           memory guard (kills the studio below 4 GB MemAvailable)
scripts/start_guard.sh        start the guard detached
scripts/get_models.sh         download Qwen-Image-2.1 + MiniMax H3 (Comfy-Org)
scripts/push_studio.sh        copy the studio to another Spark over the fabric
scripts/studio_status.py      one line per studio + the DeepSeek endpoint
examples/graphs.py            ComfyUI API graphs: Qwen turbo/normal, H3 text-to-video
examples/studio_dispatch.py   one-job-per-Spark dispatcher with the memory checks
bench/bench4x.py              the four-at-once benchmark (studios + DeepSeek decode probe)
docs/MEMORY.md                the memory budget in detail
docs/SETUP.md                 ComfyUI, Python packages, model files
docs/TWO-SPARKS.md            the 2-Spark variant (not built yet)
```

## Credits

See [CREDITS.md](CREDITS.md). Our scripts are MIT ([LICENSE](LICENSE)).
