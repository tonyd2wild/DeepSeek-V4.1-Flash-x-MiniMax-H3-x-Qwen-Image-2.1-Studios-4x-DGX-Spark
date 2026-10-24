#!/usr/bin/env python3
"""studio_dispatch.py: hand picture and video jobs to the four Spark studios, ONE JOB PER SPARK, memory-checked.

    STUDIOS="http://spark1:8188 http://spark2:8188 http://spark3:8188 http://spark4:8188" \
      python3 studio_dispatch.py --images 4 --videos 0 --prompt "a sneaker on a concrete block"
    ... --images 0 --videos 4      four H3 clips at once (one per Spark)
    ... --images 2 --videos 2      mix: each Spark takes whatever is next in line

The rules are the ones our studio front-end runs (docs/SCHEDULING.md):
  * a Spark gets a job only when its own ComfyUI queue is empty;
  * and only when MemAvailable (ComfyUI reports it as system.ram_free) covers the job: picture 26 GB cold / 8 GB with
    the picture weights already loaded, video 37.5 GB cold / 6 GB warm, plus the 4 GB guard floor;
  * switching a Spark between Qwen and H3 calls POST /free first, so only one model family is resident;
  * after a video, POST /free (H3 leaves ~34 GB resident otherwise).
Outputs are downloaded to ./out/. Stdlib only."""
import argparse, json, os, random, sys, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphs

GUARD_GB = 4.0
NEED = {"image": (26.0, 8.0), "video": (37.5, 6.0)}   # (cold, warm) GB, measured 2026-10-23


def call(url, data=None, timeout=30):
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read() or b"{}")


class Studio:
    def __init__(self, url):
        self.url, self.loaded, self.busy = url.rstrip("/"), None, False

    def ram_free(self):
        return call(self.url + "/system_stats", timeout=8)["system"]["ram_free"] / 1e9

    def idle(self):
        q = call(self.url + "/queue", timeout=8)
        return not q.get("queue_running") and not q.get("queue_pending")

    def free(self):
        call(self.url + "/free", {"unload_models": True, "free_memory": True})
        self.loaded = None

    def can_take(self, kind):
        if self.busy or not self.idle():
            return False
        if self.loaded and self.loaded != kind:
            self.free(); time.sleep(3)
        cold, warm = NEED[kind]
        return self.ram_free() >= (warm if self.loaded == kind else cold) + GUARD_GB

    def run(self, kind, prompt, out_dir):
        seed = random.randint(1, 2**31)
        g = graphs.qwen(prompt, seed) if kind == "image" else graphs.h3(prompt, seed)
        t0 = time.time()
        pid = call(self.url + "/prompt", {"prompt": g, "client_id": "studio_dispatch"})["prompt_id"]
        while True:
            time.sleep(2)
            h = call(self.url + "/history/" + pid, timeout=10)
            if pid in h and (h[pid].get("status", {}).get("completed") or h[pid].get("status", {}).get("status_str") == "error"):
                break
        self.loaded = kind
        for node in h[pid].get("outputs", {}).values():
            for key in ("images", "videos", "gifs"):
                for f in node.get(key, []) or []:
                    q = "filename=%s&subfolder=%s&type=%s" % (f["filename"], f.get("subfolder", ""), f.get("type", "output"))
                    data = urllib.request.urlopen(self.url + "/view?" + q, timeout=120).read()
                    path = os.path.join(out_dir, "%s_%s" % (self.url.split("//")[1].split(":")[0], f["filename"]))
                    open(path, "wb").write(data)
                    print("  %s: %s done in %.0f s -> %s" % (self.url, kind, time.time() - t0, path))
        if kind == "video":
            self.free()
        self.busy = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--prompt", default="A studio product photo of a white and green basketball sneaker on a concrete block")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    urls = os.environ.get("STUDIOS", "").split()
    if not urls:
        sys.exit("set STUDIOS to the space-separated ComfyUI URLs, one per Spark")
    os.makedirs(a.out, exist_ok=True)
    studios = [Studio(u) for u in urls]
    line = ["video"] * a.videos + ["image"] * a.images
    threads = []
    while line:
        for s in studios:
            if line and s.can_take(line[0]):
                s.busy = True
                t = threading.Thread(target=s.run, args=(line.pop(0), a.prompt, a.out)); t.start(); threads.append(t)
        time.sleep(3)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
