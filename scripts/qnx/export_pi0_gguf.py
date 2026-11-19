#!/usr/bin/env python3
"""Convert pi0 (openpi PyTorch port) to GGUF for the ggml implementation (pi0-ggml), and write a reference set.

  <out>/pi0-<type>.gguf   weights + hyperparameters (the vocab table is not included: language embeddings are an input)
  <out>/ref/              raw little-endian inputs and PyTorch outputs + manifest.txt, read by `pi0-run --ref`

--size tiny   gemma "dummy" variants + a shrunken SigLIP (as export_pi0_onnx.py --size tiny)
--size bench  real pi0 widths with --depth gemma/expert layers and --vision-depth SigLIP layers
--size full   gemma_2b + gemma_300m + So400m SigLIP; --weights <dir with model.safetensors> for real weights

Without --weights every 1-D parameter (norm weights, biases) is perturbed with N(0, 0.1), so the reference exercises
Gemma's (1 + w) RMSNorm scale and the biases (a fresh model has zero RMSNorm weights and zero biases).

Tensor types: 1-D tensors, the patch embedding, the position embedding and the action projections (a.*) stay F32.
Other 2-D weights use --type; Q8_0 needs the input width to be a multiple of 32, so SigLIP fc2 (4304 wide) falls
back to F16. pi0-ggml's --random mode applies the same rule.
"""
import argparse, math, os, re, types

import numpy as np
import torch

import export_pi0_onnx as E
import openpi.models_pytorch.gemma_pytorch as gp
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

import gguf

PFX = "paligemma_with_expert."
VIS = PFX + "paligemma.model.vision_tower.vision_model."
LM = PFX + "paligemma.model.language_model."
EXP = PFX + "gemma_expert.model."

GEMMA_SUB = {"input_layernorm.weight": "ln1", "post_attention_layernorm.weight": "ln2",
             "self_attn.q_proj.weight": "q", "self_attn.k_proj.weight": "k", "self_attn.v_proj.weight": "v",
             "self_attn.o_proj.weight": "o", "mlp.gate_proj.weight": "gate", "mlp.up_proj.weight": "up",
             "mlp.down_proj.weight": "down"}
VIS_SUB = {"layer_norm1": "ln1", "layer_norm2": "ln2", "self_attn.q_proj": "q", "self_attn.k_proj": "k",
           "self_attn.v_proj": "v", "self_attn.out_proj": "o", "mlp.fc1": "fc1", "mlp.fc2": "fc2"}
ACT = {"action_in_proj": "a.in", "action_out_proj": "a.out", "state_proj": "a.state",
       "action_time_mlp_in": "a.tin", "action_time_mlp_out": "a.tout"}


def gguf_name(k: str):
    """torch state_dict key -> GGUF tensor name, or None to skip (vocab tables, the unused prefix final norm)."""
    if k.startswith(VIS):
        r = k[len(VIS):]
        if r.startswith("embeddings.patch_embedding."):
            return "v.patch." + r.rsplit(".", 1)[1][0]  # w / b
        if r == "embeddings.position_embedding.weight":
            return "v.pos"
        if r.startswith("post_layernorm."):
            return "v.post." + r.rsplit(".", 1)[1][0]
        m = re.match(r"encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$", r)
        if m and m.group(2) in VIS_SUB:
            return f"v.{m.group(1)}.{VIS_SUB[m.group(2)]}.{m.group(3)[0]}"
    elif k.startswith(PFX + "paligemma.model.multi_modal_projector.linear."):
        return "mm." + k.rsplit(".", 1)[1][0]
    elif k.startswith(LM) or k.startswith(EXP):
        p, r = ("g", k[len(LM):]) if k.startswith(LM) else ("e", k[len(EXP):])
        if r == "norm.weight":
            return None if p == "g" else "e.norm"  # prefix final norm: only K/V leave the prefix
        if r.startswith("embed_tokens"):
            return None
        m = re.match(r"layers\.(\d+)\.(.+)$", r)
        if m and m.group(2) in GEMMA_SUB:
            return f"{p}.{m.group(1)}.{GEMMA_SUB[m.group(2)]}"
    else:
        mod, kind = k.rsplit(".", 1)
        if mod in ACT:
            return f"{ACT[mod]}.{kind[0]}"
    if "lm_head" in k or "embed_tokens" in k:
        return None
    raise KeyError(f"unmapped parameter {k}")


def is_gemma_norm(name):
    return re.match(r"[ge]\.\d+\.ln[12]$", name) is not None or name == "e.norm"


def pick_type(name, shape, want):
    """Shared rule with pi0-ggml --random (keep in sync with pi0.cpp pick_type)."""
    if len(shape) != 2 or name in ("v.pos",) or name.startswith("a."):
        return gguf.GGMLQuantizationType.F32
    if want == "f32":
        return gguf.GGMLQuantizationType.F32
    if want == "q8_0" and shape[-1] % 32 == 0:
        return gguf.GGMLQuantizationType.Q8_0
    return gguf.GGMLQuantizationType.F16


def write_gguf(path, m, want, meta):
    w = gguf.GGUFWriter(path, "pi0")
    for k, v in meta.items():
        (w.add_float32 if isinstance(v, float) else w.add_uint32)(f"pi0.{k}", v)
    counts = {}
    for k, t in m.state_dict().items():
        name = gguf_name(k)
        if name is None:
            continue
        a = t.detach().to(torch.float32).numpy()
        if is_gemma_norm(name):
            a = 1.0 + a  # Gemma RMSNorm scales by (1 + w); store the factor
        ty = pick_type(name, a.shape, want)
        if ty == gguf.GGMLQuantizationType.Q8_0:
            w.add_tensor(name, gguf.quants.quantize(a, ty), raw_dtype=ty)
        elif ty == gguf.GGMLQuantizationType.F16:
            w.add_tensor(name, a.astype(np.float16))
        else:
            w.add_tensor(name, np.ascontiguousarray(a))
        counts[ty.name] = counts.get(ty.name, 0) + 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return counts


def save_ref(d, **arrs):
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/manifest.txt", "w") as mf:
        for n, a in arrs.items():
            a = np.ascontiguousarray(a)
            if a.dtype == np.bool_:
                a = a.astype(np.uint8)
            a.tofile(f"{d}/{n}.bin")
            mf.write(f"{n} {a.dtype} {' '.join(map(str, a.shape))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["tiny", "bench", "full"], default="tiny")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--vision-depth", type=int, default=2)
    ap.add_argument("--weights")
    ap.add_argument("--type", choices=["f32", "f16", "q8_0"], action="append",
                    help="GGUF types to write (repeatable); default f32 and q8_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cams", default="1,1,0", help="img_masks per camera (pi0_libero pads the third)")
    ap.add_argument("--lang-len", type=int, default=48)
    ap.add_argument("--lang-valid", type=int, default=12)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--config", default="pi0_libero", choices=sorted(E.CONFIG_HORIZON))
    a = ap.parse_args()
    types_ = a.type or ["f32", "q8_0"]
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    if a.size == "tiny":
        gp.PaliGemmaForConditionalGeneration = E.tiny_vision(gp.PaliGemmaForConditionalGeneration)
        variants = ("dummy", "dummy")
    else:
        if a.size == "bench":
            E.bench_patches(a.depth, a.vision_depth)
        variants = ("gemma_2b", "gemma_300m")
    cfg = types.SimpleNamespace(pi05=False, paligemma_variant=variants[0], action_expert_variant=variants[1],
                                dtype="float32", action_dim=32, action_horizon=E.CONFIG_HORIZON[a.config],
                                pytorch_compile_mode=None)
    m = PI0Pytorch(cfg).eval()
    if a.weights:
        from safetensors.torch import load_file
        missing, unexpected = m.load_state_dict(load_file(os.path.join(a.weights, "model.safetensors")), strict=False)
        print("weights loaded; missing", len(missing), "unexpected", len(unexpected))
    else:
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.ndim == 1:
                    p.add_(torch.randn_like(p) * 0.1)

    pg = m.paligemma_with_expert.paligemma.config
    vc, tc = pg.vision_config, pg.text_config
    ec = m.paligemma_with_expert.gemma_expert.config
    meta = {"vision.width": vc.hidden_size, "vision.depth": vc.num_hidden_layers, "vision.heads": vc.num_attention_heads,
            "vision.mlp": vc.intermediate_size, "vision.patch": vc.patch_size, "vision.image": vc.image_size,
            "vision.eps": float(vc.layer_norm_eps),
            "vlm.width": tc.hidden_size, "vlm.depth": tc.num_hidden_layers, "vlm.heads": tc.num_attention_heads,
            "vlm.kv_heads": tc.num_key_value_heads, "vlm.head_dim": tc.head_dim, "vlm.mlp": tc.intermediate_size,
            "expert.width": ec.hidden_size, "expert.depth": ec.num_hidden_layers, "expert.heads": ec.num_attention_heads,
            "expert.kv_heads": ec.num_key_value_heads, "expert.head_dim": ec.head_dim, "expert.mlp": ec.intermediate_size,
            "rms_eps": float(tc.rms_norm_eps), "rope_theta": float(tc.rope_theta),
            "action_dim": cfg.action_dim, "action_horizon": cfg.action_horizon, "num_steps": a.steps,
            "time_min_period": E.TIME_MIN_PERIOD, "time_max_period": E.TIME_MAX_PERIOD}
    print("model", a.size, variants, {k: v for k, v in meta.items() if "depth" in k or "width" in k})

    # Reference: same inputs and maths as export_pi0_onnx.py, with partial camera / language masks
    B, L, W = 1, a.lang_len, tc.hidden_size
    cams = [c == "1" for c in a.cams.split(",")]
    images = torch.rand(B, len(cams), 3, vc.image_size, vc.image_size) * 2 - 1
    img_masks = torch.tensor([cams])
    tokens = torch.randint(0, tc.vocab_size, (B, L))
    lang_masks = torch.zeros(B, L, dtype=torch.bool); lang_masks[:, :a.lang_valid] = True
    state = torch.randn(B, cfg.action_dim)
    noise = torch.randn(B, cfg.action_horizon, cfg.action_dim)
    ew = ec.hidden_size
    pe, ds = E.PrefixEncoder(m).eval(), E.DenoiseStep(m).eval()
    with torch.no_grad():
        lang_emb = m.paligemma_with_expert.embed_language_tokens(tokens) * math.sqrt(W)
        img0 = m.paligemma_with_expert.embed_image(images[:, 0])
        pre = pe(images, img_masks, lang_emb, lang_masks)
        pad, kv = pre[0], pre[1:]
        v1 = ds(state, pad, noise, E.time_embedding(torch.ones(B), ew), *kv)
        x, tt = noise.clone(), torch.tensor(1.0, dtype=torch.float32)
        dt = torch.tensor(-1.0 / a.steps, dtype=torch.float32)
        while tt >= -dt / 2:
            x = x + dt * ds(state, pad, x, E.time_embedding(tt.expand(B), ew), *kv)
            tt += dt
    depth = tc.num_hidden_layers
    kvd = {f"{kind}_{i}": kv[2 * i + (kind == "v")][0, 0].numpy() for i in range(depth) for kind in ("k", "v")}
    save_ref(f"{a.out}/ref", images=images[0].numpy(), img_masks=img_masks[0].numpy(), lang_emb=lang_emb[0].numpy(),
             lang_masks=lang_masks[0].numpy(), state=state[0].numpy(), noise=noise[0].numpy(),
             prefix_pad_masks=pad[0].numpy(), img0_emb=img0[0].numpy(), v_t_at_t1=v1[0].numpy(),
             actions=x[0].numpy(), **kvd)
    print(f"reference: P={pad.shape[1]} valid={int(pad.sum())} kv {tuple(kv[0].shape)} actions {tuple(x.shape)}")
    for t in types_:
        c = write_gguf(f"{a.out}/pi0-{t}.gguf", m, t, meta)
        print(f"pi0-{t}.gguf {os.path.getsize(f'{a.out}/pi0-{t}.gguf')/1e6:.1f} MB", c)


if __name__ == "__main__":
    main()
