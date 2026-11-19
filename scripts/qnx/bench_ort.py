#!/usr/bin/env python3
"""Time prefix_encoder and denoise_step (10 steps) separately; compare actions to the fp32 torch reference.
usage: bench_ort.py <dir> [runs] [threads]"""
import json, sys, time
import numpy as np, onnxruntime as ort
d = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5; thr = int(sys.argv[3]) if len(sys.argv) > 3 else 0
meta = json.load(open(f"{d}/meta.json")); ref = np.load(f"{d}/reference.npz")
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if thr: so.intra_op_num_threads = thr
t = time.monotonic()
sp = ort.InferenceSession(f"{d}/prefix_encoder.onnx", so, providers=["CPUExecutionProvider"])
sd = ort.InferenceSession(f"{d}/denoise_step.onnx", so, providers=["CPUExecutionProvider"])
load_s = time.monotonic() - t
kvn = [o.name for o in sp.get_outputs()][1:]
W, lo, hi = meta["expert_width"], meta["time_min_period"], meta["time_max_period"]
def te(t):
    f = np.linspace(0, 1, W // 2); x = (2 * np.pi / (lo * (hi / lo) ** f))[None] * t
    return np.concatenate([np.sin(x), np.cos(x)], 1).astype(np.float32)
P, D = [], []
for r in range(runs + 1):
    t0 = time.monotonic(); po = sp.run(None, {k: ref[k] for k in ("images", "img_masks", "lang_emb", "lang_masks")}); t1 = time.monotonic()
    feed = {"state": ref["state"], "prefix_pad_masks": po[0], **dict(zip(kvn, po[1:]))}
    x, tt, dt = ref["noise"].copy(), np.float32(1), np.float32(-0.1)
    while tt >= -dt / 2:
        feed["x_t"], feed["time_emb"] = x, te(float(tt)); x = x + dt * sd.run(None, feed)[0]; tt = np.float32(tt + dt)
    t2 = time.monotonic()
    if r: P.append((t1 - t0) * 1e3); D.append((t2 - t1) * 1e3)
err = float(np.abs(x - ref["actions"]).max()); scale = float(np.abs(ref["actions"]).max())
print(json.dumps({"dir": d.split("/")[-1], "depth": meta["depth"], "vision_depth": meta["vision_depth"], "threads": thr,
                  "load_s": round(load_s, 2), "prefix_ms": round(sorted(P)[len(P)//2], 1),
                  "denoise10_ms": round(sorted(D)[len(D)//2], 1), "actions_maxerr": round(err, 5), "actions_maxabs": round(scale, 3)}))
