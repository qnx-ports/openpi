#!/usr/bin/env python3
"""Run the exported pi0 graphs with ONNX Runtime and compare with the PyTorch reference.

usage: python3 validate_ort.py <dir with prefix_encoder.onnx, denoise_step.onnx, reference.npz, meta.json> [runs]
"""
import json, sys, time
import numpy as np
import onnxruntime as ort

d = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
meta = json.load(open(f"{d}/meta.json"))
ref = np.load(f"{d}/reference.npz")
print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
print("model", meta["size"], meta["variants"], f"params {meta['params']/1e6:.1f}M", "prefix_len", meta["prefix_len"])

so = ort.SessionOptions()
t0 = time.time()
sp = ort.InferenceSession(f"{d}/prefix_encoder.onnx", so, providers=["CPUExecutionProvider"])
sd = ort.InferenceSession(f"{d}/denoise_step.onnx", so, providers=["CPUExecutionProvider"])
print(f"sessions loaded in {time.time()-t0:.2f}s")
kv_names = [o.name for o in sp.get_outputs()][1:]

def time_embedding(t, width):
    # as openpi create_sinusoidal_pos_embedding: float64 maths, float32 result
    frac = np.linspace(0.0, 1.0, width // 2, dtype=np.float64)
    period = meta["time_min_period"] * (meta["time_max_period"] / meta["time_min_period"]) ** frac
    x = (1.0 / period * 2 * np.pi)[None, :] * np.array([t], np.float64)[:, None]
    return np.concatenate([np.sin(x), np.cos(x)], axis=1).astype(np.float32)


def sample():
    tp = time.time()
    po = sp.run(None, {k: ref[k] for k in ("images", "img_masks", "lang_emb", "lang_masks")})
    tp = time.time() - tp
    kv = dict(zip(kv_names, po[1:]))
    x, t, dt = ref["noise"].copy(), 1.0, -1.0 / meta["steps"]
    td = time.time()
    while t >= -dt / 2:
        v = sd.run(None, {"state": ref["state"], "prefix_pad_masks": po[0], "x_t": x,
                          "time_emb": time_embedding(t, meta["expert_width"]), **kv})[0]
        x = x + dt * v
        t += dt
    return po, x, tp, time.time() - td

po, x, _, _ = sample()  # warm-up + correctness
kv_err = max(float(np.abs(po[1 + i] - ref[n]).max()) for i, n in enumerate(kv_names))
act_err = float(np.abs(x - ref["actions"]).max())
print(f"max |kv - torch| = {kv_err:.3g}   max |actions - torch| = {act_err:.3g}")
ok = kv_err < 1e-3 and act_err < 1e-3
times = [sample()[2:] for _ in range(runs)]
tp = sorted(a for a, _ in times)[len(times) // 2] * 1000
td = sorted(b for _, b in times)[len(times) // 2] * 1000
print(f"median over {runs}: prefix {tp:.1f} ms + {meta['steps']} denoise steps {td:.1f} ms = {tp+td:.1f} ms")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
