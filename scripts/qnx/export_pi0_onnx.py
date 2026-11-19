#!/usr/bin/env python3
"""Export pi0 (PyTorch port) as two ONNX graphs for ONNX Runtime on QNX.

  prefix_encoder.onnx  images + image masks + language embeddings  -> prefix pad mask + per-layer K/V cache
  denoise_step.onnx    state + prefix pad mask + K/V cache + x_t + time_emb -> v_t   (called num_steps times)
                       time_emb = sinusoid(t) computed on the host in float64 (see time_embedding / meta.json)

Language embeddings (token lookup * sqrt(width)) are an INPUT, so the 257k-row vocab table never ships to the
target. The PaliGemma transformer itself still runs in prefix_encoder: image and language tokens attend to each
other there, so it cannot be precomputed per task.

--size tiny  : gemma "dummy" variants + a shrunken SigLIP (same 224px / patch 14 -> 256 tokens per camera), for
               building and validating the pipeline. Random weights. Real tensor shapes except widths.
--size full  : gemma_2b + gemma_300m + So400m SigLIP (needs ~30+ GB RAM to export; weights via --weights).

Also writes reference.npz: fixed random inputs and the PyTorch outputs for both graphs plus a full 10-step
Euler sample, so the target can check its numbers against PyTorch.
"""
import argparse, dataclasses, json, math, os, shutil, time, types

import numpy as np
import torch
from torch import nn

import openpi.models_pytorch.gemma_pytorch as gp
import openpi.models_pytorch.pi0_pytorch as pi0mod
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks, create_sinusoidal_pos_embedding

# action_horizon of the openpi TrainConfigs the QNX server has profiles for (openpi 215abfb training/config.py)
CONFIG_HORIZON = {"pi0_libero": 50, "pi0_aloha": 50, "pi0_aloha_sim": 50, "pi0_droid": 10}

# Timestep embedding constants used by PI0Pytorch.embed_suffix (openpi 215abfb)
TIME_MIN_PERIOD, TIME_MAX_PERIOD = 4e-3, 4.0


def time_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    """Host-side: float64 sinusoid exactly as upstream, then cast to float32 (upstream does .type(timestep.dtype))."""
    return create_sinusoidal_pos_embedding(t, width, TIME_MIN_PERIOD, TIME_MAX_PERIOD, device=t.device).to(torch.float32)

NUM_CAMS, IMG = 3, 224


def tiny_vision(orig_cls):
    """Wrap PaliGemmaForConditionalGeneration so the vision tower is tiny and projects to the text width."""
    def build(config):
        v = config.vision_config
        v.hidden_size, v.num_hidden_layers, v.num_attention_heads, v.intermediate_size = 64, 2, 4, 128
        v.projection_dim = config.text_config.hidden_size
        config.projection_dim = config.text_config.hidden_size
        config.hidden_size = config.text_config.hidden_size
        return orig_cls(config=config)
    return build


def bench_patches(depth: int, vision_depth: int, vocab: int = 1024):
    """Real pi0 layer widths (gemma_2b / gemma_300m / So400m SigLIP) with only `depth` transformer layers, `vision_depth`
    SigLIP layers and a tiny vocab (the embedding table is not part of either graph). Random weights; for timing and
    memory extrapolation only."""
    import openpi.models.gemma as _g
    orig_get, orig_pg, orig_ge = _g.get_config, gp.PaliGemmaForConditionalGeneration, gp.GemmaForCausalLM

    def get_config(variant):
        return dataclasses.replace(orig_get(variant), depth=depth)

    def pg(config):
        config.vision_config.num_hidden_layers = vision_depth
        config.text_config.vocab_size = vocab
        return orig_pg(config=config)

    def ge(config):
        config.vocab_size = vocab
        return orig_ge(config=config)

    _g.get_config, gp.PaliGemmaForConditionalGeneration, gp.GemmaForCausalLM = get_config, pg, ge


class PrefixEncoder(nn.Module):
    def __init__(self, m: PI0Pytorch):
        super().__init__()
        self.m = m

    def forward(self, images, img_masks, lang_emb, lang_masks):
        # images: (B, NUM_CAMS, 3, 224, 224) float in [-1, 1]; img_masks: (B, NUM_CAMS) bool
        # lang_emb: (B, L, W) = embed_tokens(tokens) * sqrt(W); lang_masks: (B, L) bool
        pwe = self.m.paligemma_with_expert
        embs, pads = [], []
        for c in range(images.shape[1]):
            e = pwe.embed_image(images[:, c])
            embs.append(e)
            pads.append(img_masks[:, c : c + 1].expand(e.shape[0], e.shape[1]))
        embs.append(lang_emb)
        pads.append(lang_masks)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pads, dim=1)
        # full attention among image + language tokens; int (not bool) because ONNX CumSum rejects bool
        att_masks = torch.zeros_like(pad_masks, dtype=torch.int64)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        pos = torch.cumsum(pad_masks.to(torch.int64), dim=1) - 1
        lm = pwe.paligemma.language_model
        lm.config._attn_implementation = "eager"
        out = lm.forward(inputs_embeds=embs, attention_mask=self.m._prepare_attention_masks_4d(att_2d),
                         position_ids=pos, past_key_values=None, use_cache=True)
        kv = out.past_key_values.to_legacy_cache()
        flat = [t for layer in kv for t in layer]  # k0, v0, k1, v1, ...
        return (pad_masks, *flat)


class DenoiseStep(nn.Module):
    def __init__(self, m: PI0Pytorch):
        super().__init__()
        self.m = m

    def forward(self, state, prefix_pad_masks, x_t, time_emb, *kv):
        # time_emb (B, W) is computed on the host (time_embedding above): upstream builds it in float64, and ORT 1.24's
        # CPU provider has no float64 Sin/Cos. Feed it to embed_suffix by standing in for the sinusoid function.
        orig = pi0mod.create_sinusoidal_pos_embedding
        pi0mod.create_sinusoidal_pos_embedding = lambda *a, **k: time_emb
        try:
            return self._forward(state, prefix_pad_masks, x_t, torch.zeros(time_emb.shape[0]), kv)
        finally:
            pi0mod.create_sinusoidal_pos_embedding = orig

    def _forward(self, state, prefix_pad_masks, x_t, t, kv):
        # Same maths as PI0Pytorch.denoise_step (openpi 215abfb), with bool masks cast to int before CumSum/Sum,
        # which ONNX does not define for bool.
        from transformers.cache_utils import DynamicCache
        m = self.m
        past = DynamicCache.from_legacy_cache(tuple((kv[i], kv[i + 1]) for i in range(0, len(kv), 2)))
        suffix_embs, suffix_pad, suffix_att, adarms_cond = m.embed_suffix(state, x_t, t)
        bsz, slen, plen = prefix_pad_masks.shape[0], suffix_pad.shape[1], prefix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(bsz, slen, plen)
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_4d = m._prepare_attention_masks_4d(torch.cat([prefix_2d, suffix_2d], dim=2))
        offsets = torch.sum(prefix_pad_masks.to(torch.int64), dim=-1)[:, None]
        pos = offsets + torch.cumsum(suffix_pad.to(torch.int64), dim=1) - 1
        m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        out, _ = m.paligemma_with_expert.forward(attention_mask=full_4d, position_ids=pos, past_key_values=past,
                                                 inputs_embeds=[None, suffix_embs], use_cache=False,
                                                 adarms_cond=[None, adarms_cond])
        return m.action_out_proj(out[1][:, -m.config.action_horizon :].to(torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["tiny", "full", "bench"], default="tiny")
    ap.add_argument("--depth", type=int, default=1, help="bench: gemma/expert layers (real pi0: 18)")
    ap.add_argument("--vision-depth", type=int, default=1, help="bench: SigLIP layers (real pi0: 27)")
    ap.add_argument("--weights", help="dir with model.safetensors (full size only); default: random weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--lang-len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="pi0_libero", choices=sorted(CONFIG_HORIZON),
                    help="openpi TrainConfig the bundle serves (sets action_horizon and the server profile)")
    ap.add_argument("--norm-stats", help="norm_stats.json from the checkpoint assets (default: synthetic, tiny only)")
    ap.add_argument("--embed-dtype", choices=["f32", "bf16"], default="f32",
                    help="storage of embed_tokens.npy (bf16 = raw bits; exact for bf16 checkpoints, half the size)")
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model"))
    ap.add_argument("--save-torch", action="store_true", help="also save torch_model.pt (Linux-side equivalence tests only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    if a.size == "tiny":
        gp.PaliGemmaForConditionalGeneration = tiny_vision(gp.PaliGemmaForConditionalGeneration)
        variants = ("dummy", "dummy")
    elif a.size == "bench":
        bench_patches(a.depth, a.vision_depth)
        variants = ("gemma_2b", "gemma_300m")
    else:
        variants = ("gemma_2b", "gemma_300m")
    cfg = types.SimpleNamespace(pi05=False, paligemma_variant=variants[0], action_expert_variant=variants[1],
                                dtype="float32", action_dim=32, action_horizon=CONFIG_HORIZON[a.config],
                                pytorch_compile_mode=None)
    m = PI0Pytorch(cfg).eval()
    if a.weights:
        from safetensors.torch import load_file
        missing, unexpected = m.load_state_dict(load_file(os.path.join(a.weights, "model.safetensors")), strict=False)
        print("weights loaded; missing", len(missing), "unexpected", len(unexpected))
    nparams = sum(p.numel() for p in m.parameters())
    width = m.paligemma_with_expert.paligemma.config.text_config.hidden_size
    depth = m.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
    print(f"model: {a.size} {variants} params={nparams/1e6:.1f}M width={width} depth={depth}")

    B, L = 1, a.lang_len
    images = torch.rand(B, NUM_CAMS, 3, IMG, IMG) * 2 - 1
    img_masks = torch.ones(B, NUM_CAMS, dtype=torch.bool)
    tokens = torch.randint(0, m.paligemma_with_expert.paligemma.config.text_config.vocab_size, (B, L))
    lang_masks = torch.zeros(B, L, dtype=torch.bool); lang_masks[:, :12] = True  # a 12-token prompt, rest padding
    with torch.no_grad():
        lang_emb = m.paligemma_with_expert.embed_language_tokens(tokens) * math.sqrt(width)
    state = torch.randn(B, 32)
    noise = torch.randn(B, cfg.action_horizon, cfg.action_dim)
    t1 = torch.ones(B)
    ew = m.action_in_proj.out_features

    pe, ds = PrefixEncoder(m).eval(), DenoiseStep(m).eval()
    with torch.no_grad():
        pre = pe(images, img_masks, lang_emb, lang_masks)
        pad, kv = pre[0], pre[1:]
        v1 = ds(state, pad, noise, time_embedding(t1, ew), *kv)
        # full Euler sample exactly as PI0Pytorch.sample_actions: time and dt are float32 tensors
        x, tt = noise.clone(), torch.tensor(1.0, dtype=torch.float32)
        dt = torch.tensor(-1.0 / a.steps, dtype=torch.float32)
        while tt >= -dt / 2:
            x = x + dt * ds(state, pad, x, time_embedding(tt.expand(B), ew), *kv)
            tt += dt
    P = pad.shape[1]
    print(f"prefix tokens P={P}  kv tensors={len(kv)} each {tuple(kv[0].shape)}  v_t {tuple(v1.shape)}")

    kv_names = [f"{kind}_{i}" for i in range(depth) for kind in ("k", "v")]
    big = nparams * 4 > 1.8e9
    t0 = time.time()
    torch.onnx.export(pe, (images, img_masks, lang_emb, lang_masks), f"{a.out}/prefix_encoder.onnx",
                      input_names=["images", "img_masks", "lang_emb", "lang_masks"],
                      output_names=["prefix_pad_masks", *kv_names], opset_version=a.opset, dynamo=False,
                      external_data=big)
    torch.onnx.export(ds, (state, pad, noise, time_embedding(t1, ew), *kv), f"{a.out}/denoise_step.onnx",
                      input_names=["state", "prefix_pad_masks", "x_t", "time_emb", *kv_names],
                      output_names=["v_t"], opset_version=a.opset, dynamo=False, external_data=big)
    print(f"exported in {time.time()-t0:.1f}s")

    np.savez(f"{a.out}/reference.npz", images=images.numpy(), img_masks=img_masks.numpy(),
             lang_emb=lang_emb.numpy(), lang_masks=lang_masks.numpy(), state=state.numpy(), noise=noise.numpy(),
             prefix_pad_masks=pad.numpy(), v_t_at_t1=v1.numpy(), actions=x.numpy(),
             **{n: t.numpy() for n, t in zip(kv_names, kv)})
    meta = dict(size=a.size, variants=variants, params=nparams, width=width, depth=depth, prefix_len=P,
                lang_len=L, num_cams=NUM_CAMS, action_horizon=cfg.action_horizon, action_dim=cfg.action_dim,
                steps=a.steps, opset=a.opset, expert_width=ew,
                time_min_period=TIME_MIN_PERIOD, time_max_period=TIME_MAX_PERIOD, kv_shape=list(kv[0].shape), torch=torch.__version__)
    meta.update(openpi_config=a.config, embed_dtype=a.embed_dtype, openpi_commit="215abfb")
    if a.size == "bench":
        meta.update(depth=a.depth, vision_depth=a.vision_depth)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    if a.size == "bench":  # timing artefacts only: no serving bundle
        _ort_selfcheck(a, B, ew, kv_names, dt=None)
        return

    # --- serving bundle: embedding table, tokenizer, norm stats
    table = m.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.detach().to(torch.float32)
    if a.embed_dtype == "bf16":
        table = table.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    else:
        table = table.numpy()
    np.save(f"{a.out}/embed_tokens.npy", table)
    shutil.copyfile(a.tokenizer, f"{a.out}/paligemma_tokenizer.model")
    if a.norm_stats:
        shutil.copyfile(a.norm_stats, f"{a.out}/norm_stats.json")
    elif a.size == "tiny":
        # Synthetic stats (clearly not real): exercise normalisation with non-trivial values
        r = np.random.default_rng(a.seed)
        dims = {"pi0_libero": (8, 7), "pi0_droid": (8, 8), "pi0_aloha": (14, 14), "pi0_aloha_sim": (14, 14)}[a.config]
        def st(n):
            return {"mean": r.normal(0, 0.5, n).tolist(), "std": r.uniform(0.2, 2.0, n).tolist(),
                    "q01": (-r.uniform(1, 2, n)).tolist(), "q99": r.uniform(1, 2, n).tolist()}
        json.dump({"norm_stats": {"state": st(dims[0]), "actions": st(dims[1]), "_synthetic": None}},
                  open(f"{a.out}/norm_stats.json", "w"), indent=1)
        # drop the marker key (openpi's loader would reject it) but keep the file honest via meta
        d = json.load(open(f"{a.out}/norm_stats.json")); d["norm_stats"].pop("_synthetic")
        json.dump(d, open(f"{a.out}/norm_stats.json", "w"), indent=1)
        meta["norm_stats"] = "synthetic"; json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    else:
        raise SystemExit("--norm-stats is required for --size full")
    if a.save_torch:
        torch.save(m.state_dict(), f"{a.out}/torch_model.pt")  # tied lm_head/embed weights: safetensors refuses them

    _ort_selfcheck(a, B, ew, kv_names, dt=None)


def _ort_selfcheck(a, B, ew, kv_names, dt):
    """Self-check with ONNX Runtime on this host."""
    import onnxruntime as ort
    ref = np.load(f"{a.out}/reference.npz")
    sp = ort.InferenceSession(f"{a.out}/prefix_encoder.onnx", providers=["CPUExecutionProvider"])
    sd = ort.InferenceSession(f"{a.out}/denoise_step.onnx", providers=["CPUExecutionProvider"])
    po = sp.run(None, {k: ref[k] for k in ("images", "img_masks", "lang_emb", "lang_masks")})
    kvd = dict(zip(kv_names, po[1:]))
    x = ref["noise"].copy(); tt = np.float32(1.0); dtn = np.float32(-1.0 / a.steps)
    while tt >= -dtn / np.float32(2):
        te = time_embedding(torch.tensor([float(tt)], dtype=torch.float32), ew).numpy()
        v = sd.run(None, {"state": ref["state"], "prefix_pad_masks": po[0], "x_t": x, "time_emb": te, **kvd})[0]
        x = x + dtn * v; tt = np.float32(tt + dtn)
    print("ORT vs torch: max|kv0| diff", float(np.abs(po[1] - ref["k_0"]).max()),
          " max|actions| diff", float(np.abs(x - ref["actions"]).max()))
    print("inputs:", [i.name for i in sd.get_inputs()][:6], "...")


if __name__ == "__main__":
    main()
