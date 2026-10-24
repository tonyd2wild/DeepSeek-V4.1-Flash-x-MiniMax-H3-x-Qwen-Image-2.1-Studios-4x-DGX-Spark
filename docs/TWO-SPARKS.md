# Two Sparks (not built yet)

The same idea should work on a 2-Spark setup: DeepSeek tensor-parallel across 2 Sparks plus a studio on each. **We have
not built or measured it.** This page is the arithmetic, so nobody mistakes it for a result.

## Why the current TP2 lane does not leave room

Our validated TP2 lane ([DeepSeek recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks),
LibertAI EXL3 2.9 bpw, 1M context, 4 streams, KV pool 1,048,576) puts about 105 GiB on each Spark. MemAvailable while it
serves is 6.7 and 8.2 GiB. A studio needs about 26 GB free to start a picture and about 37.5 GB to start a video
(docs/MEMORY.md), so it does not fit.

## What would have to change

Roughly 25-35 GB more free memory per Spark, from some mix of:

- **a smaller quant** of DeepSeek-V4.1-Flash (fewer bits per weight on the experts);
- **a smaller KV pool / fewer streams** (the pool is allocated up front, so it is a direct trade);
- **pictures only** (Qwen-Image-2.1 needs much less than H3).

Each of these costs something (quality, context, concurrency). Which trade is worth it needs a real run with evals, the
same way the TP4 and TP2 lanes were validated.

If you try it, please open an issue with your numbers.
