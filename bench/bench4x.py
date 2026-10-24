#!/usr/bin/env python3
"""4-at-once co-tenancy bench: DeepSeek TP4 serving while all 4 Sparks render (Qwen x4, then H3 x4).
Measures per-node render wall time, min MemAvailable (ComfyUI system.ram_free), and DeepSeek single-stream
decode tok/s before / during each phase. Uses only ComfyUI's HTTP API (+ /free), same as Spark Studio."""
import json, sys, threading, time, urllib.request, random, os
OUT = os.path.dirname(os.path.abspath(__file__))
# STUDIOS="http://spark1:8188 http://spark2:8188 http://spark3:8188 http://spark4:8188" DEEPSEEK=http://spark1:8000
SPARKS = {"s%d" % (i + 1): u for i, u in enumerate(os.environ.get("STUDIOS", "").split())}
DS = os.environ.get("DEEPSEEK", "http://spark1:8000") + "/v1/chat/completions"
def req(url, data=None, timeout=30):
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read() or b"{}")
def ram_free(u): return req(u + "/system_stats", timeout=8)["system"]["ram_free"] / 1e9
def free(u): req(u + "/free", {"unload_models": True, "free_memory": True}, timeout=30)
SIG = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"
def qwen_turbo(seed):
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "A studio product photo of a white and green basketball sneaker on a concrete block, soft rim light, 85mm"}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1328, "height": 1328, "batch_size": 1}},
            "11": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["4", 0]}},
            "12": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIG}},
            "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "8": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["13", 0], "sigmas": ["12", 0], "latent_image": ["6", 0]}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "bench4x/IMG"}}}
def h3_t2v(seed, w=960, h=544, length=56, steps=20):
    return {"6": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
            "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
            "11": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
            "24": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
            "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "width": w, "height": h, "length": length,
                    "prompt": "Cinematic. A sneaker reviewer in a bright studio lifts a white and green basketball sneaker to the camera and says: 'Four Sparks, one model, four studios.' [Shot 1] 0.0s to 2.3s, one continuous shot, slow push in. Constraints: every frame sharp and detailed. overall_soundscape: quiet studio room tone. non_diegetic_music: N/A."}},
            "16": {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": ["104", 0]}},
            "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
            "9": {"class_type": "BasicScheduler", "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
            "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
            "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
            "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24.0}},
            "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": "bench4x/VID", "format": "auto", "codec": "auto"}}}

phase = {"name": "idle"}; stop = threading.Event()
memlog = []; dslog = []
def mem_sampler():
    while not stop.is_set():
        row = {"t": time.time(), "phase": phase["name"]}
        for n, u in SPARKS.items():
            try: row[n] = round(ram_free(u), 2)
            except Exception: row[n] = None
        memlog.append(row); time.sleep(1)
def ds_probe():
    while not stop.is_set():
        t0 = time.time(); ph = phase["name"]
        try:
            r = req(DS, {"model": "deepseek-v4.1-flash", "max_tokens": 200, "temperature": 0.7, "reasoning_effort": "none",
                         "messages": [{"role": "user", "content": "Write about 150 words on why sneakers get re-released."}]}, timeout=180)
            el = time.time() - t0; n = r["usage"]["completion_tokens"]
            dslog.append({"t": t0, "phase": ph, "tokens": n, "secs": round(el, 2), "tps": round(n / el, 1)})
        except Exception as e:
            dslog.append({"t": t0, "phase": ph, "error": str(e)[:120]})
def run_phase(name, graph_fn):
    phase["name"] = name; t0 = time.time(); ids = {}
    for n, u in SPARKS.items():
        ids[n] = req(u + "/prompt", {"prompt": graph_fn(random.randint(1, 2**31)), "client_id": "bench4x"})["prompt_id"]
    done = {}; outs = {}
    while len(done) < len(ids) and time.time() - t0 < 1800:
        for n, pid in ids.items():
            if n in done: continue
            try:
                h = req(SPARKS[n] + "/history/" + pid, timeout=10)
                if pid in h:
                    st = h[pid].get("status", {})
                    if st.get("completed") or st.get("status_str") in ("success", "error"):
                        done[n] = {"secs": round(time.time() - t0, 1), "status": st.get("status_str")}
                        outs[n] = h[pid].get("outputs", {})
            except Exception as e:
                done.setdefault(n + "_err", str(e)[:80])
        time.sleep(2)
    return {"wall": round(time.time() - t0, 1), "nodes": done, "outputs": outs}
def save_outputs(res, tag):
    for n, outs in res["outputs"].items():
        for node in outs.values():
            for key in ("images", "gifs", "videos", "animated"):
                for f in node.get(key, []) if isinstance(node.get(key), list) else []:
                    if not isinstance(f, dict) or "filename" not in f: continue
                    q = "filename=%s&subfolder=%s&type=%s" % (f["filename"], f.get("subfolder", ""), f.get("type", "output"))
                    try:
                        data = urllib.request.urlopen(SPARKS[n] + "/view?" + q, timeout=120).read()
                        open(os.path.join(OUT, "%s_%s_%s" % (tag, n, f["filename"])), "wb").write(data)
                    except Exception as e: print("fetch fail", n, e)
if __name__ == "__main__":
    if len(SPARKS) < 1: sys.exit("set STUDIOS (and DEEPSEEK)")
    # ASK FIRST on shared hardware: this loads every studio at once and slows the LLM while it runs.
    for n, u in SPARKS.items():
        q = req(u + "/queue", timeout=8)
        if q.get("queue_running") or q.get("queue_pending"): sys.exit("busy: " + n)
    for u in SPARKS.values(): free(u)
    time.sleep(8)
    threading.Thread(target=mem_sampler, daemon=True).start()
    threading.Thread(target=ds_probe, daemon=True).start()
    phase["name"] = "baseline"; time.sleep(45)
    res = {"qwen4": run_phase("qwen4", qwen_turbo)}
    save_outputs(res["qwen4"], "qwen4")
    phase["name"] = "between"
    for u in SPARKS.values(): free(u)
    time.sleep(20)
    res["h3x4"] = run_phase("h3x4", h3_t2v)
    save_outputs(res["h3x4"], "h3x4")
    phase["name"] = "after"
    for u in SPARKS.values(): free(u)
    time.sleep(30); stop.set(); time.sleep(3)
    for k in res: res[k].pop("outputs", None)
    summary = {}
    for ph in ("baseline", "qwen4", "between", "h3x4", "after"):
        tps = [d["tps"] for d in dslog if d.get("phase") == ph and "tps" in d]
        mins = {n: min([r[n] for r in memlog if r["phase"] == ph and r.get(n) is not None] or [None]) for n in SPARKS}
        summary[ph] = {"ds_tps": tps, "ds_tps_avg": round(sum(tps) / len(tps), 1) if tps else None, "min_free_gb": mins}
    json.dump({"results": res, "summary": summary, "dslog": dslog, "memlog": memlog}, open(os.path.join(OUT, "bench4x.json"), "w"), indent=1)
    print(json.dumps({"results": res, "summary": summary}, indent=1))
