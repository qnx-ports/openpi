"""QnxPolicy: openpi Policy.infer() semantics on top of the ONNX graphs (no torch/jax on the target)."""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

from . import transforms as T
from .model import Pi0Onnx
from .prompt import PromptEncoder


class QnxPolicy:
    def __init__(self, model_dir: str, profile: str | None = None, default_prompt: str | None = None,
                 threads: int = 0, seed: int | None = None):
        self.model = Pi0Onnx(model_dir, threads=threads)
        meta = self.model.meta
        name = profile or meta.get("openpi_config")
        if name not in T.PROFILES:
            raise ValueError(f"unknown profile {name!r}; known: {sorted(T.PROFILES)}")
        self.profile = T.PROFILES[name]
        if self.profile.action_horizon != self.model.action_horizon:
            raise ValueError(f"profile {name} expects action_horizon {self.profile.action_horizon}, "
                             f"graphs were exported with {self.model.action_horizon}")
        if self.model.num_cams != len(T.IMAGE_KEYS):
            raise ValueError(f"graphs take {self.model.num_cams} cameras, pi0 uses {len(T.IMAGE_KEYS)}")
        self.default_prompt = default_prompt if default_prompt is not None else self.profile.default_prompt
        self.norm_stats = T.load_norm_stats(os.path.join(model_dir, "norm_stats.json"))
        self.prompts = PromptEncoder(model_dir, meta)
        self.rng = np.random.default_rng(seed)
        self._metadata = {"server": "openpi-qnx", "openpi_config": name,
                          "model": {k: meta[k] for k in ("size", "variants", "params", "action_horizon", "action_dim")
                                    if k in meta}}

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        p, ah, ad = self.profile, self.model.action_horizon, self.model.action_dim
        data = copy.copy(obs)
        if self.default_prompt is not None and "prompt" not in data:
            data["prompt"] = self.default_prompt
        data = p.inputs(data)  # DeltaActions on inputs is a no-op without "actions"
        data = T.normalize(data, self.norm_stats, p.quantile_norm)
        data["image"] = T.resize_images(data["image"])
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = np.asarray(prompt).item()
        state = T.pad_to_dim(np.asarray(data["state"], np.float32), ad).astype(np.float32)

        # Observation.from_dict: uint8 -> float32 in [-1, 1]; the graphs take CHW
        imgs, masks = [], []
        for k in T.IMAGE_KEYS:
            im = np.asarray(data["image"][k])
            if im.dtype == np.uint8:
                im = im.astype(np.float32) / 255.0 * 2.0 - 1.0
            imgs.append(np.transpose(im.astype(np.float32), (2, 0, 1)))
            masks.append(bool(data["image_mask"][k]))
        images = np.stack(imgs)[None]
        img_masks = np.asarray(masks, bool)[None]
        lang_emb, lang_masks = self.prompts.encode(prompt)

        if noise is not None and np.ndim(noise) == 2:
            noise = np.asarray(noise)[None]
        t0 = time.monotonic()
        actions, timing = self.model.sample(images, img_masks, lang_emb, lang_masks, state[None], noise=noise,
                                            rng=self.rng)
        model_ms = (time.monotonic() - t0) * 1000

        out = {"state": state, "actions": actions.reshape(ah, ad)}
        out = T.unnormalize(out, self.norm_stats, p.quantile_norm)
        out = T.absolute_actions(out, p.delta_mask)
        out = p.outputs(out)
        out["policy_timing"] = {"infer_ms": model_ms, **timing}
        return out
