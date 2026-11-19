"""pi0 inference with ONNX Runtime: prefix_encoder once per observation, then N Euler steps of denoise_step.

The graphs and meta.json come from export_pi0_onnx.py. Everything here is numpy + onnxruntime; no torch or jax.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import onnxruntime as ort


class Pi0Onnx:
    def __init__(self, model_dir: str, threads: int = 0, providers: list[str] | None = None):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            so.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        self._prefix = ort.InferenceSession(os.path.join(model_dir, "prefix_encoder.onnx"), so, providers=providers)
        self._denoise = ort.InferenceSession(os.path.join(model_dir, "denoise_step.onnx"), so, providers=providers)
        self._kv_names = [o.name for o in self._prefix.get_outputs()][1:]
        self.action_horizon = int(self.meta["action_horizon"])
        self.action_dim = int(self.meta["action_dim"])
        self.num_cams = int(self.meta["num_cams"])
        self.lang_len = int(self.meta["lang_len"])
        self.steps = int(self.meta["steps"])

    def time_embedding(self, t: float) -> np.ndarray:
        """openpi create_sinusoidal_pos_embedding: float64 maths, float32 result. Shape (1, expert_width)."""
        width = int(self.meta["expert_width"])
        lo, hi = float(self.meta["time_min_period"]), float(self.meta["time_max_period"])
        frac = np.linspace(0.0, 1.0, width // 2, dtype=np.float64)
        period = lo * (hi / lo) ** frac
        x = (1.0 / period * 2 * np.pi)[None, :] * np.array([t], np.float64)[:, None]
        return np.concatenate([np.sin(x), np.cos(x)], axis=1).astype(np.float32)

    def sample(self, images, img_masks, lang_emb, lang_masks, state, noise=None, num_steps=None, rng=None):
        """images (1,C,3,224,224) f32 in [-1,1]; img_masks (1,C) bool; lang_emb (1,L,W) f32; lang_masks (1,L) bool;
        state (1,action_dim) f32 (normalised, padded). Returns actions (action_horizon, action_dim) f32 and timings."""
        steps = num_steps or self.steps
        if noise is None:
            rng = rng or np.random.default_rng()
            noise = rng.standard_normal((1, self.action_horizon, self.action_dim)).astype(np.float32)
        noise = np.asarray(noise, np.float32).reshape(1, self.action_horizon, self.action_dim)

        t0 = time.monotonic()
        out = self._prefix.run(None, {"images": images, "img_masks": img_masks, "lang_emb": lang_emb,
                                      "lang_masks": lang_masks})
        t1 = time.monotonic()
        feed = {"state": state, "prefix_pad_masks": out[0], **dict(zip(self._kv_names, out[1:]))}
        # PI0Pytorch.sample_actions keeps time and dt as float32 tensors and accumulates in float32; do the same so the
        # timesteps (0.9, 0.8, ... as float32 sums, not exact decimals) match upstream bit for bit.
        x, t, dt = noise, np.float32(1.0), np.float32(-1.0 / steps)
        while t >= -dt / np.float32(2):
            feed["x_t"], feed["time_emb"] = x, self.time_embedding(float(t))
            v = self._denoise.run(None, feed)[0]
            x = x + dt * v
            t = np.float32(t + dt)
        t2 = time.monotonic()
        return x[0], {"prefix_ms": (t1 - t0) * 1000, "denoise_ms": (t2 - t1) * 1000}
