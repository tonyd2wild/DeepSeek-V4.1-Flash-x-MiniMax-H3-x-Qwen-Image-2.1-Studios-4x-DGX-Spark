#!/usr/bin/env python3
"""studio_status.py: one line per Spark studio (MemAvailable, queue, ComfyUI version) plus the DeepSeek endpoint.
    STUDIOS="http://spark1:8188 ..." DEEPSEEK=http://spark1:8000 python3 studio_status.py"""
import json, os, urllib.request
def get(u, t=8):
    with urllib.request.urlopen(u, timeout=t) as f: return json.loads(f.read())
for u in os.environ.get("STUDIOS", "").split():
    try:
        s = get(u + "/system_stats")["system"]; q = get(u + "/queue")
        print("%-28s free %5.1f GB of %5.1f   running %d  queued %d   ComfyUI %s" % (
            u, s["ram_free"] / 1e9, s["ram_total"] / 1e9, len(q["queue_running"]), len(q["queue_pending"]), s.get("comfyui_version")))
    except Exception as e:
        print("%-28s DOWN (%s)" % (u, e))
ds = os.environ.get("DEEPSEEK")
if ds:
    try:
        h = get(ds + "/health")
        print("%-28s %s  streams %s/%s waiting %s" % (ds, h.get("status"), h.get("live", {}).get("connections"),
                                                     h.get("max_batch_size"), h.get("live", {}).get("waiting")))
    except Exception as e:
        print("%-28s DOWN (%s)" % (ds, e))
