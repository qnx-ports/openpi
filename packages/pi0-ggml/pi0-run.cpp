// pi0-run: Physical Intelligence pi0 (openpi PyTorch port, commit 215abfb) inference with ggml.
//
//   prefix   SigLIP over the valid camera images + Gemma 2B over [image tokens, language tokens] -> per-layer K/V
//   denoise  Gemma 300M action expert over [state token, 50 action tokens] attending to the prefix K/V; 10 Euler steps
//
// Padded prefix tokens (cameras with img_mask = 0, language padding) are dropped instead of masked. openpi masks them
// out of every attention row of a valid token, and their RoPE positions (cumsum(pad) - 1) equal the compacted index,
// so the valid outputs are unchanged.
//
// Weights: -m pi0-<type>.gguf (export_pi0_gguf.py) or --random tiny|full (in-memory random weights, for timing).
// Check:   --ref <dir> compares K/V, the first image embedding, v_t at t=1 and the final actions with PyTorch.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define DIE(...) do { fprintf(stderr, "pi0-run: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

static const int GRAPH_SIZE = 16384;

struct gemma_hp { int width, depth, heads, kv_heads, head_dim, mlp; };

struct hparams {
    int v_width, v_depth, v_heads, v_mlp, v_patch, v_image;
    float v_eps;
    gemma_hp g, e;
    float rms_eps, rope_theta;
    int action_dim, horizon, steps;
    float tmin, tmax;
    int n_patches() const { return (v_image / v_patch) * (v_image / v_patch); }
};

static hparams preset(const std::string & name) {
    hparams h{};
    h.v_patch = 14; h.v_image = 224; h.v_eps = 1e-6f; h.rms_eps = 1e-6f; h.rope_theta = 10000.0f;
    h.action_dim = 32; h.horizon = 50; h.steps = 10; h.tmin = 4e-3f; h.tmax = 4.0f;
    if (name == "full") {  // gemma_2b + gemma_300m + So400m/14 (openpi models/gemma.py, PaliGemma defaults)
        h.v_width = 1152; h.v_depth = 27; h.v_heads = 16; h.v_mlp = 4304;
        h.g = {2048, 18, 8, 1, 256, 16384};
        h.e = {1024, 18, 8, 1, 256, 4096};
    } else if (name == "tiny") {  // gemma "dummy" + export_pi0_onnx.py tiny_vision
        h.v_width = 64; h.v_depth = 2; h.v_heads = 4; h.v_mlp = 128;
        h.g = {64, 4, 8, 1, 16, 128};
        h.e = {64, 4, 8, 1, 16, 128};
    } else {
        DIE("unknown preset %s (tiny|full)", name.c_str());
    }
    return h;
}

// ---------------------------------------------------------------------------------------------- tensor inventory

struct tspec { std::string name; std::vector<int64_t> ne; char kind; };  // kind: w matrix, b bias, n norm, l layernorm w

static std::vector<tspec> inventory(const hparams & h) {
    std::vector<tspec> s;
    const int64_t V = h.v_width, M = h.v_mlp, ps = h.v_patch;
    s.push_back({"v.patch.w", {ps, ps, 3, V}, 'w'});
    s.push_back({"v.patch.b", {V}, 'b'});
    s.push_back({"v.pos", {V, h.n_patches()}, 'w'});
    for (int i = 0; i < h.v_depth; i++) {
        const std::string p = "v." + std::to_string(i) + ".";
        for (const char * n : {"ln1", "ln2"}) { s.push_back({p + n + ".w", {V}, 'l'}); s.push_back({p + n + ".b", {V}, 'b'}); }
        for (const char * n : {"q", "k", "v", "o"}) { s.push_back({p + n + ".w", {V, V}, 'w'}); s.push_back({p + n + ".b", {V}, 'b'}); }
        s.push_back({p + "fc1.w", {V, M}, 'w'}); s.push_back({p + "fc1.b", {M}, 'b'});
        s.push_back({p + "fc2.w", {M, V}, 'w'}); s.push_back({p + "fc2.b", {V}, 'b'});
    }
    s.push_back({"v.post.w", {V}, 'l'}); s.push_back({"v.post.b", {V}, 'b'});
    s.push_back({"mm.w", {V, h.g.width}, 'w'}); s.push_back({"mm.b", {h.g.width}, 'b'});
    for (int pass = 0; pass < 2; pass++) {
        const gemma_hp & g = pass ? h.e : h.g;
        const int64_t W = g.width, Q = g.heads * g.head_dim, KV = g.kv_heads * g.head_dim, F = g.mlp;
        for (int i = 0; i < g.depth; i++) {
            const std::string p = std::string(pass ? "e." : "g.") + std::to_string(i) + ".";
            s.push_back({p + "ln1", {W}, 'n'}); s.push_back({p + "ln2", {W}, 'n'});
            s.push_back({p + "q", {W, Q}, 'w'}); s.push_back({p + "k", {W, KV}, 'w'}); s.push_back({p + "v", {W, KV}, 'w'});
            s.push_back({p + "o", {Q, W}, 'w'});
            s.push_back({p + "gate", {W, F}, 'w'}); s.push_back({p + "up", {W, F}, 'w'}); s.push_back({p + "down", {F, W}, 'w'});
        }
    }
    const int64_t E = h.e.width, A = h.action_dim;
    s.push_back({"e.norm", {E}, 'n'});
    s.push_back({"a.in.w", {A, E}, 'w'}); s.push_back({"a.in.b", {E}, 'b'});
    s.push_back({"a.out.w", {E, A}, 'w'}); s.push_back({"a.out.b", {A}, 'b'});
    s.push_back({"a.state.w", {A, E}, 'w'}); s.push_back({"a.state.b", {E}, 'b'});
    s.push_back({"a.tin.w", {2 * E, E}, 'w'}); s.push_back({"a.tin.b", {E}, 'b'});
    s.push_back({"a.tout.w", {E, E}, 'w'}); s.push_back({"a.tout.b", {E}, 'b'});
    return s;
}

// Same rule as export_pi0_gguf.py pick_type
static ggml_type pick_type(const tspec & t, ggml_type want) {
    if (t.ne.size() != 2 || t.name == "v.pos" || t.name.rfind("a.", 0) == 0 || want == GGML_TYPE_F32) return GGML_TYPE_F32;
    if (want == GGML_TYPE_Q8_0 && t.ne[0] % 32 == 0) return GGML_TYPE_Q8_0;
    return GGML_TYPE_F16;
}

// ---------------------------------------------------------------------------------------------- model

struct model {
    hparams hp{};
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t(const std::string & n) const {
        ggml_tensor * x = ggml_get_tensor(ctx, n.c_str());
        if (!x) DIE("missing tensor %s", n.c_str());
        return x;
    }
};

static void check_inventory(const model & m) {
    for (const tspec & s : inventory(m.hp)) {
        ggml_tensor * x = m.t(s.name);
        for (size_t d = 0; d < 4; d++) {
            const int64_t want = d < s.ne.size() ? s.ne[d] : 1;
            if (x->ne[d] != want) DIE("tensor %s: ne[%zu]=%lld, expected %lld", s.name.c_str(), d, (long long)x->ne[d], (long long)want);
        }
    }
}

static void load_gguf(model & m, const char * path, ggml_backend_t backend) {
    ggml_context * meta = nullptr;
    gguf_init_params p = {true, &meta};
    gguf_context * g = gguf_init_from_file(path, p);
    if (!g) DIE("cannot read %s", path);
    auto key = [&](const char * k) {
        const int64_t id = gguf_find_key(g, (std::string("pi0.") + k).c_str());
        if (id < 0) DIE("%s: missing key pi0.%s", path, k);
        return id;
    };
    auto u = [&](const char * k) { return (int)gguf_get_val_u32(g, key(k)); };
    auto f = [&](const char * k) { return gguf_get_val_f32(g, key(k)); };
    hparams & h = m.hp;
    h.v_width = u("vision.width"); h.v_depth = u("vision.depth"); h.v_heads = u("vision.heads"); h.v_mlp = u("vision.mlp");
    h.v_patch = u("vision.patch"); h.v_image = u("vision.image"); h.v_eps = f("vision.eps");
    for (int pass = 0; pass < 2; pass++) {
        const std::string b = pass ? "expert." : "vlm.";
        gemma_hp & x = pass ? h.e : h.g;
        x.width = u((b + "width").c_str()); x.depth = u((b + "depth").c_str()); x.heads = u((b + "heads").c_str());
        x.kv_heads = u((b + "kv_heads").c_str()); x.head_dim = u((b + "head_dim").c_str()); x.mlp = u((b + "mlp").c_str());
    }
    h.rms_eps = f("rms_eps"); h.rope_theta = f("rope_theta");
    h.action_dim = u("action_dim"); h.horizon = u("action_horizon"); h.steps = u("num_steps");
    h.tmin = f("time_min_period"); h.tmax = f("time_max_period");

    m.ctx = meta;
    m.buf = ggml_backend_alloc_ctx_tensors(meta, backend);
    if (!m.buf) DIE("cannot allocate weights on %s", ggml_backend_name(backend));
    FILE * fp = fopen(path, "rb");
    if (!fp) DIE("cannot open %s", path);
    std::vector<uint8_t> tmp(64u << 20);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        ggml_tensor * x = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i), n = ggml_nbytes(x);
        if (fseeko(fp, (off_t)off, SEEK_SET) != 0) DIE("seek failed in %s", path);
        for (size_t pos = 0; pos < n;) {
            const size_t c = std::min(tmp.size(), n - pos);
            if (fread(tmp.data(), 1, c, fp) != c) DIE("short read in %s", path);
            ggml_backend_tensor_set(x, tmp.data(), pos, c);
            pos += c;
        }
    }
    fclose(fp);
    gguf_free(g);
    check_inventory(m);
}

static void init_random(model & m, ggml_type want, ggml_backend_t backend, uint64_t seed) {
    const std::vector<tspec> inv = inventory(m.hp);
    ggml_init_params ip = {ggml_tensor_overhead() * (inv.size() + 8), nullptr, true};
    m.ctx = ggml_init(ip);
    std::vector<ggml_type> types;
    for (const tspec & s : inv) {
        const ggml_type ty = pick_type(s, want);
        ggml_tensor * x = ggml_new_tensor(m.ctx, ty, (int)s.ne.size(), s.ne.data());
        ggml_set_name(x, s.name.c_str());
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, backend);
    if (!m.buf) DIE("cannot allocate weights on %s", ggml_backend_name(backend));
    uint64_t r = seed * 6364136223846793005ull + 1442695040888963407ull;
    auto rnd = [&]() { r = r * 6364136223846793005ull + 1442695040888963407ull; return (float)((r >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; };
    std::vector<float> f32;
    std::vector<uint8_t> out;
    for (const tspec & s : inv) {
        ggml_tensor * x = m.t(s.name);
        const int64_t row = x->ne[0], nrows = ggml_nelements(x) / row;
        const int64_t chunk = std::max<int64_t>(1, (16 << 20) / row);
        const float a = s.kind == 'w' ? 0.0346f : 0.17f;  // uniform: std 0.02 for matrices, 0.1 for 1-D
        const float base = (s.kind == 'n' || s.kind == 'l') ? 1.0f : 0.0f;
        for (int64_t r0 = 0; r0 < nrows; r0 += chunk) {
            const int64_t nr = std::min(chunk, nrows - r0);
            f32.resize(nr * row);
            for (float & v : f32) v = base + a * rnd();
            const size_t rb = ggml_row_size(x->type, row);
            out.resize(nr * rb);
            if (x->type == GGML_TYPE_F32) memcpy(out.data(), f32.data(), out.size());
            else if (x->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *)out.data(), nr * row);
            else ggml_quantize_chunk(x->type, f32.data(), out.data(), 0, nr, row, nullptr);
            ggml_backend_tensor_set(x, out.data(), r0 * rb, out.size());
        }
    }
    check_inventory(m);
}

// ---------------------------------------------------------------------------------------------- graph pieces

static ggml_tensor * linear(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b = nullptr) {
    x = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, x, b) : x;
}

static ggml_tensor * layer_norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), w), b);
}

// Gemma RMSNorm; the GGUF stores (1 + w)
static ggml_tensor * rms_norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// q (hd, nh, T, N), k/v (hd, nkv, S, N), mask (S, T) or null  ->  (hd*nh, T, N)
static ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, float scale) {
    const int64_t hd = q->ne[0], nh = q->ne[1], T = q->ne[2], N = q->ne[3];
    q = ggml_permute(c, q, 0, 2, 1, 3);                                     // (hd, T, nh, N)
    k = ggml_permute(c, k, 0, 2, 1, 3);                                     // (hd, S, nkv, N)
    ggml_tensor * kq = ggml_mul_mat(c, k, q);                               // (S, T, nh, N)
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(c, kq, mask, scale, 0.0f);
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));        // (S, hd, nkv, N)
    ggml_tensor * kqv = ggml_mul_mat(c, vt, kq);                            // (hd, T, nh, N)
    kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));                   // (hd, nh, T, N)
    return ggml_reshape_3d(c, kqv, hd * nh, T, N);
}

static ggml_tensor * rope(ggml_context * c, const hparams & h, ggml_tensor * x, ggml_tensor * pos) {
    return ggml_rope_ext(c, x, pos, nullptr, (int)x->ne[0], GGML_ROPE_TYPE_NEOX, 0, h.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

struct kv_cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;  // (head_dim, kv_heads, P_max) F32
    int p_max = 0;
};

static void kv_init(kv_cache & kv, const hparams & h, int p_max, ggml_backend_t backend) {
    ggml_init_params ip = {ggml_tensor_overhead() * (2 * h.g.depth + 4), nullptr, true};
    kv.ctx = ggml_init(ip);
    kv.p_max = p_max;
    for (int l = 0; l < h.g.depth; l++) {
        kv.k.push_back(ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, h.g.head_dim, h.g.kv_heads, p_max));
        kv.v.push_back(ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, h.g.head_dim, h.g.kv_heads, p_max));
    }
    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
}

static ggml_tensor * kv_view(ggml_context * c, ggml_tensor * t, int P) {
    return ggml_view_3d(c, t, t->ne[0], t->ne[1], P, t->nb[1], t->nb[2], 0);
}

struct graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::map<std::string, ggml_tensor *> in, out;
    void free() { if (ctx) ggml_free(ctx); ctx = nullptr; }
};

static void graph_begin(graph & g) {
    ggml_init_params ip = {ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false), nullptr, true};
    g.ctx = ggml_init(ip);
    g.gf = ggml_new_graph_custom(g.ctx, GRAPH_SIZE, false);
}

static ggml_tensor * input(graph & g, const char * name, ggml_tensor * t) {
    ggml_set_name(t, name); ggml_set_input(t); g.in[name] = t; return t;
}

// images (I, I, 3, ncv), lang (W, Lv)  ->  K/V written to the cache for P = ncv*np + Lv tokens
static void build_prefix(graph & g, const model & m, kv_cache & kv, int ncv, int Lv) {
    const hparams & h = m.hp;
    ggml_context * c = g.ctx;
    const int64_t V = h.v_width, np = h.n_patches(), vhd = V / h.v_heads, W = h.g.width, P = ncv * np + Lv;
    ggml_tensor * img = input(g, "images", ggml_new_tensor_4d(c, GGML_TYPE_F32, h.v_image, h.v_image, 3, ncv));
    ggml_tensor * pos = input(g, "pos", ggml_new_tensor_1d(c, GGML_TYPE_I32, P));

    ggml_tensor * x = ggml_conv_2d(c, m.t("v.patch.w"), img, h.v_patch, h.v_patch, 0, 0, 1, 1);  // (16, 16, V, ncv)
    x = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, x, np, V, ncv), 1, 0, 2, 3));         // (V, np, ncv)
    x = ggml_add(c, ggml_add(c, x, m.t("v.patch.b")), m.t("v.pos"));
    for (int l = 0; l < h.v_depth; l++) {
        const std::string p = "v." + std::to_string(l) + ".";
        ggml_tensor * y = layer_norm(c, x, m.t(p + "ln1.w"), m.t(p + "ln1.b"), h.v_eps);
        ggml_tensor * q = ggml_reshape_4d(c, linear(c, y, m.t(p + "q.w"), m.t(p + "q.b")), vhd, h.v_heads, np, ncv);
        ggml_tensor * k = ggml_reshape_4d(c, linear(c, y, m.t(p + "k.w"), m.t(p + "k.b")), vhd, h.v_heads, np, ncv);
        ggml_tensor * v = ggml_reshape_4d(c, linear(c, y, m.t(p + "v.w"), m.t(p + "v.b")), vhd, h.v_heads, np, ncv);
        y = attention(c, q, k, v, nullptr, 1.0f / sqrtf((float)vhd));
        x = ggml_add(c, x, linear(c, y, m.t(p + "o.w"), m.t(p + "o.b")));
        y = layer_norm(c, x, m.t(p + "ln2.w"), m.t(p + "ln2.b"), h.v_eps);
        y = linear(c, ggml_gelu(c, linear(c, y, m.t(p + "fc1.w"), m.t(p + "fc1.b"))), m.t(p + "fc2.w"), m.t(p + "fc2.b"));
        x = ggml_add(c, x, y);
    }
    x = layer_norm(c, x, m.t("v.post.w"), m.t("v.post.b"), h.v_eps);
    x = linear(c, x, m.t("mm.w"), m.t("mm.b"));                                                 // (W, np, ncv)
    ggml_set_name(x, "img_emb"); ggml_set_output(x); g.out["img_emb"] = x;
    x = ggml_reshape_2d(c, x, W, np * ncv);
    if (Lv > 0) x = ggml_concat(c, x, input(g, "lang", ggml_new_tensor_2d(c, GGML_TYPE_F32, W, Lv)), 1);  // (W, P)

    const gemma_hp & gh = h.g;
    for (int l = 0; l < gh.depth; l++) {
        const std::string p = "g." + std::to_string(l) + ".";
        ggml_tensor * y = rms_norm(c, x, m.t(p + "ln1"), h.rms_eps);
        ggml_tensor * q = rope(c, h, ggml_reshape_3d(c, linear(c, y, m.t(p + "q")), gh.head_dim, gh.heads, P), pos);
        ggml_tensor * k = rope(c, h, ggml_reshape_3d(c, linear(c, y, m.t(p + "k")), gh.head_dim, gh.kv_heads, P), pos);
        ggml_tensor * v = ggml_reshape_3d(c, linear(c, y, m.t(p + "v")), gh.head_dim, gh.kv_heads, P);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, k, kv_view(c, kv.k[l], P)));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, v, kv_view(c, kv.v[l], P)));
        if (l == gh.depth - 1) break;  // only the last layer's K/V leave the prefix
        y = attention(c, q, k, v, nullptr, 1.0f / sqrtf((float)gh.head_dim));
        x = ggml_add(c, x, linear(c, y, m.t(p + "o")));
        y = rms_norm(c, x, m.t(p + "ln2"), h.rms_eps);
        y = ggml_geglu_split(c, linear(c, y, m.t(p + "gate")), linear(c, y, m.t(p + "up")));
        x = ggml_add(c, x, linear(c, y, m.t(p + "down")));
    }
    ggml_build_forward_expand(g.gf, g.out["img_emb"]);
}

// state (A), x_t (A, H), time_emb (E), pos (1+H), mask (P+1+H, 1+H)  ->  v_t (A, H)
static void build_denoise(graph & g, const model & m, kv_cache & kv, int P) {
    const hparams & h = m.hp;
    ggml_context * c = g.ctx;
    const gemma_hp & e = h.e;
    const int64_t A = h.action_dim, H = h.horizon, S = 1 + H, E = e.width;
    ggml_tensor * state = input(g, "state", ggml_new_tensor_2d(c, GGML_TYPE_F32, A, 1));
    ggml_tensor * xt = input(g, "x_t", ggml_new_tensor_2d(c, GGML_TYPE_F32, A, H));
    ggml_tensor * temb = input(g, "time_emb", ggml_new_tensor_2d(c, GGML_TYPE_F32, E, 1));
    ggml_tensor * pos = input(g, "pos", ggml_new_tensor_1d(c, GGML_TYPE_I32, S));
    ggml_tensor * mask = input(g, "mask", ggml_new_tensor_2d(c, GGML_TYPE_F32, P + S, S));

    ggml_tensor * se = linear(c, state, m.t("a.state.w"), m.t("a.state.b"));                    // (E, 1)
    ggml_tensor * ae = linear(c, xt, m.t("a.in.w"), m.t("a.in.b"));                             // (E, H)
    ggml_tensor * at = ggml_concat(c, ae, ggml_repeat(c, temb, ae), 0);                         // (2E, H)
    at = linear(c, ggml_silu(c, linear(c, at, m.t("a.tin.w"), m.t("a.tin.b"))), m.t("a.tout.w"), m.t("a.tout.b"));
    ggml_tensor * x = ggml_concat(c, se, at, 1);                                                // (E, S)
    for (int l = 0; l < e.depth; l++) {
        const std::string p = "e." + std::to_string(l) + ".";
        ggml_tensor * y = rms_norm(c, x, m.t(p + "ln1"), h.rms_eps);
        ggml_tensor * q = rope(c, h, ggml_reshape_3d(c, linear(c, y, m.t(p + "q")), e.head_dim, e.heads, S), pos);
        ggml_tensor * k = rope(c, h, ggml_reshape_3d(c, linear(c, y, m.t(p + "k")), e.head_dim, e.kv_heads, S), pos);
        ggml_tensor * v = ggml_reshape_3d(c, linear(c, y, m.t(p + "v")), e.head_dim, e.kv_heads, S);
        k = ggml_concat(c, kv_view(c, kv.k[l], P), k, 2);                                        // (hd, nkv, P+S)
        v = ggml_concat(c, kv_view(c, kv.v[l], P), v, 2);
        y = attention(c, q, k, v, mask, 1.0f / sqrtf((float)e.head_dim));
        x = ggml_add(c, x, linear(c, y, m.t(p + "o")));
        y = rms_norm(c, x, m.t(p + "ln2"), h.rms_eps);
        y = ggml_geglu_split(c, linear(c, y, m.t(p + "gate")), linear(c, y, m.t(p + "up")));
        x = ggml_add(c, x, linear(c, y, m.t(p + "down")));
    }
    x = rms_norm(c, x, m.t("e.norm"), h.rms_eps);
    x = ggml_view_2d(c, x, E, H, x->nb[1], x->nb[1]);                                           // action tokens
    ggml_tensor * vt = linear(c, x, m.t("a.out.w"), m.t("a.out.b"));
    ggml_set_name(vt, "v_t"); ggml_set_output(vt); g.out["v_t"] = vt;
    ggml_build_forward_expand(g.gf, vt);
}

// create_sinusoidal_pos_embedding: float64 maths, float32 result
static void time_embedding(const hparams & h, float t, int width, float * out) {
    const int half = width / 2;
    for (int i = 0; i < half; i++) {
        const double frac = half > 1 ? (double)i / (half - 1) : 0.0;
        const double period = (double)h.tmin * pow((double)h.tmax / (double)h.tmin, frac);
        const double x = 1.0 / period * 2.0 * M_PI * (double)t;
        out[i] = (float)sin(x);
        out[half + i] = (float)cos(x);
    }
}

// ---------------------------------------------------------------------------------------------- reference files

struct array { std::string dtype; std::vector<int64_t> shape; std::vector<uint8_t> data;
    const float * f() const { return (const float *)data.data(); }
    size_t n() const { size_t k = 1; for (auto s : shape) k *= s; return k; } };

static std::map<std::string, array> load_ref(const std::string & dir) {
    std::map<std::string, array> r;
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) DIE("cannot read %s/manifest.txt", dir.c_str());
    std::string line;
    while (std::getline(mf, line)) {
        std::istringstream ss(line);
        std::string name; array a; ss >> name >> a.dtype;
        int64_t d; while (ss >> d) a.shape.push_back(d);
        std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
        a.data.assign(std::istreambuf_iterator<char>(f), {});
        r[name] = a;
    }
    return r;
}

struct cmp { double max_diff = 0, max_ref = 0; size_t nonfinite = 0; };
static void cmp_add(cmp & c, const float * got, const float * ref, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(got[i])) { c.nonfinite++; continue; }  // std::max would silently drop NaN
        c.max_diff = std::max(c.max_diff, (double)fabsf(got[i] - ref[i]));
        c.max_ref = std::max(c.max_ref, (double)fabsf(ref[i]));
    }
}
static void cmp_print(const char * what, const cmp & c) {
    printf("  %-14s max|diff| %.3e   max|ref| %.3e   rel %.3e%s\n", what, c.max_diff, c.max_ref,
           c.max_ref > 0 ? c.max_diff / c.max_ref : 0.0, c.nonfinite ? "   NON-FINITE VALUES" : "");
    if (c.nonfinite) printf("  %-14s %zu non-finite outputs\n", "", c.nonfinite);
}

// ---------------------------------------------------------------------------------------------- main

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void usage() {
    fprintf(stderr,
        "usage: pi0-run (-m pi0.gguf | --random tiny|full [--type q8_0|f16|f32] [--depth N] [--vision-depth N])\n"
        "               [--ref DIR] [--device gpu|cpu] [-t THREADS] [--runs N] [--cams N] [--lang N]\n"
        "  --ref DIR   inputs and PyTorch outputs from export_pi0_gguf.py; prints the differences\n"
        "  --cams/--lang  valid cameras / language tokens for random inputs (default 3 / 48: P = 816)\n");
    exit(2);
}

int main(int argc, char ** argv) {
    std::string mpath, random, ref_dir, device = "gpu";
    ggml_type want = GGML_TYPE_Q8_0;
    int threads = 0, runs = 1, cams = 3, lang = 48, depth = -1, vdepth = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) usage(); return std::string(argv[++i]); };
        if (a == "-m") mpath = next();
        else if (a == "--random") random = next();
        else if (a == "--type") { std::string t = next(); want = t == "f32" ? GGML_TYPE_F32 : t == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0; }
        else if (a == "--depth") depth = std::stoi(next());
        else if (a == "--vision-depth") vdepth = std::stoi(next());
        else if (a == "--ref") ref_dir = next();
        else if (a == "--device") device = next();
        else if (a == "-t") threads = std::stoi(next());
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--cams") cams = std::stoi(next());
        else if (a == "--lang") lang = std::stoi(next());
        else usage();
    }
    if (mpath.empty() == random.empty()) usage();

    // CPU backend directly: touching the device registry probes Vulkan, and on a QNX VM without a GPU libkhronos
    // exits the process ("Failed to open configuration file")
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (threads > 0) ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_t gpu = device == "gpu" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : nullptr;
    if (device == "gpu" && !gpu) DIE("no GPU backend");
    ggml_backend_t primary = gpu ? gpu : cpu;
    printf("backend: %s\n", ggml_backend_name(primary));

    model m;
    auto t0 = std::chrono::steady_clock::now();
    if (!mpath.empty()) {
        load_gguf(m, mpath.c_str(), primary);
    } else {
        m.hp = preset(random);
        if (depth > 0) { m.hp.g.depth = depth; m.hp.e.depth = depth; }
        if (vdepth > 0) m.hp.v_depth = vdepth;
        init_random(m, want, primary, 1);
    }
    const hparams & h = m.hp;
    {
        std::map<std::string, std::pair<int, size_t>> by_type;
        size_t total = 0;
        for (ggml_tensor * x = ggml_get_first_tensor(m.ctx); x; x = ggml_get_next_tensor(m.ctx, x)) {
            auto & e = by_type[ggml_type_name(x->type)]; e.first++; e.second += ggml_nbytes(x); total += ggml_nbytes(x);
        }
        printf("weights: %.1f MB in %.0f ms (", total / 1e6, ms_since(t0));
        for (auto & kv : by_type) printf(" %s:%d/%.0fMB", kv.first.c_str(), kv.second.first, kv.second.second / 1e6);
        printf(" )  vision %dx%d  vlm %dx%d  expert %dx%d\n", h.v_depth, h.v_width, h.g.depth, h.g.width, h.e.depth, h.e.width);
    }

    // inputs
    std::map<std::string, array> ref;
    std::vector<float> images, lang_emb, state, noise;
    std::vector<int> valid_prefix;  // indices of valid tokens in the padded reference prefix
    const int np = h.n_patches(), W = h.g.width, A = h.action_dim, H = h.horizon;
    const size_t img_sz = (size_t)3 * h.v_image * h.v_image;
    int ncv = 0, Lv = 0;
    if (!ref_dir.empty()) {
        ref = load_ref(ref_dir);
        const array & im = ref["images"], & imm = ref["img_masks"], & le = ref["lang_emb"], & lm = ref["lang_masks"];
        for (size_t c = 0; c < imm.data.size(); c++) {
            if (!imm.data[c]) continue;
            images.insert(images.end(), im.f() + c * img_sz, im.f() + (c + 1) * img_sz);
            for (int j = 0; j < np; j++) valid_prefix.push_back((int)c * np + j);
            ncv++;
        }
        const int L = (int)lm.data.size();
        for (int j = 0; j < L; j++) {
            if (!lm.data[j]) continue;
            lang_emb.insert(lang_emb.end(), le.f() + (size_t)j * W, le.f() + (size_t)(j + 1) * W);
            valid_prefix.push_back((int)imm.data.size() * np + j);
            Lv++;
        }
        state.assign(ref["state"].f(), ref["state"].f() + A);
        noise.assign(ref["noise"].f(), ref["noise"].f() + (size_t)A * H);
    } else {
        ncv = cams; Lv = lang;
        uint64_t r = 42;
        auto rnd = [&]() { r = r * 6364136223846793005ull + 1442695040888963407ull; return (float)((r >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; };
        images.resize(img_sz * ncv); for (float & v : images) v = rnd();
        lang_emb.resize((size_t)W * Lv); for (float & v : lang_emb) v = rnd();
        state.resize(A); for (float & v : state) v = rnd();
        noise.resize((size_t)A * H); for (float & v : noise) v = rnd();
    }
    if (ncv < 1) DIE("need at least one valid camera");
    const int P = ncv * np + Lv, S = 1 + H;
    printf("prefix: %d cameras + %d language tokens = %d tokens\n", ncv, Lv, P);

    kv_cache kv;
    kv_init(kv, h, P, primary);
    std::vector<ggml_backend_t> backends = {primary};
    if (gpu) backends.push_back(cpu);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), GRAPH_SIZE, false, true);

    std::vector<float> actions, v_t1, img_emb((size_t)W * np);
    for (int run = 0; run < runs; run++) {
        // ---- prefix
        auto tp = std::chrono::steady_clock::now();
        graph gp; graph_begin(gp);
        build_prefix(gp, m, kv, ncv, Lv);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gp.gf)) DIE("prefix graph allocation failed");
        ggml_backend_tensor_set(gp.in["images"], images.data(), 0, images.size() * sizeof(float));
        if (Lv > 0) ggml_backend_tensor_set(gp.in["lang"], lang_emb.data(), 0, lang_emb.size() * sizeof(float));
        std::vector<int32_t> pos(P); for (int i = 0; i < P; i++) pos[i] = i;
        ggml_backend_tensor_set(gp.in["pos"], pos.data(), 0, pos.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(sched, gp.gf) != GGML_STATUS_SUCCESS) DIE("prefix compute failed");
        ggml_backend_tensor_get(gp.out["img_emb"], img_emb.data(), 0, img_emb.size() * sizeof(float));
        const double t_prefix = ms_since(tp);
        const int n_nodes_prefix = ggml_graph_n_nodes(gp.gf);
        gp.free();

        // ---- denoise
        auto td = std::chrono::steady_clock::now();
        graph gd; graph_begin(gd);
        build_denoise(gd, m, kv, P);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gd.gf)) DIE("denoise graph allocation failed");
        std::vector<int32_t> spos(S); for (int i = 0; i < S; i++) spos[i] = P + i;
        // state token (row 0) sees the prefix and itself; action tokens see everything
        std::vector<float> mask((size_t)(P + S) * S, 0.0f);
        for (int k = P + 1; k < P + S; k++) mask[k] = -INFINITY;
        std::vector<float> x = noise, v((size_t)A * H), te(h.e.width);
        float t = 1.0f;
        const float dt = -1.0f / h.steps;
        std::vector<double> t_steps;
        while (t >= -dt / 2) {
            auto ts = std::chrono::steady_clock::now();
            time_embedding(h, t, h.e.width, te.data());
            // every input, every step: the allocator may reuse an input's memory once the graph has consumed it
            ggml_backend_tensor_set(gd.in["pos"], spos.data(), 0, spos.size() * sizeof(int32_t));
            ggml_backend_tensor_set(gd.in["mask"], mask.data(), 0, mask.size() * sizeof(float));
            ggml_backend_tensor_set(gd.in["state"], state.data(), 0, state.size() * sizeof(float));
            ggml_backend_tensor_set(gd.in["time_emb"], te.data(), 0, te.size() * sizeof(float));
            ggml_backend_tensor_set(gd.in["x_t"], x.data(), 0, x.size() * sizeof(float));
            if (ggml_backend_sched_graph_compute(sched, gd.gf) != GGML_STATUS_SUCCESS) DIE("denoise compute failed");
            ggml_backend_tensor_get(gd.out["v_t"], v.data(), 0, v.size() * sizeof(float));
            if (t_steps.empty()) v_t1 = v;
            for (size_t i = 0; i < x.size(); i++) x[i] = x[i] + dt * v[i];
            t += dt;
            t_steps.push_back(ms_since(ts));
        }
        const double t_denoise = ms_since(td);
        gd.free();
        actions = x;
        double smin = *std::min_element(t_steps.begin(), t_steps.end());
        printf("run %d: prefix %.1f ms (%d nodes)  denoise %.1f ms (%zu steps, fastest %.1f)  total %.1f ms\n",
               run, t_prefix, n_nodes_prefix, t_denoise, t_steps.size(), smin, t_prefix + t_denoise);
    }

    if (!ref.empty()) {
        printf("vs PyTorch (%s):\n", ref_dir.c_str());
        if (ref["img_masks"].data[0]) {
            cmp c; cmp_add(c, img_emb.data(), ref["img0_emb"].f(), img_emb.size()); cmp_print("img0_emb", c);
        }
        const int hd = h.g.head_dim * h.g.kv_heads;
        std::vector<float> got((size_t)hd * P);
        cmp ck_first, ck_last, cv_last;
        for (int l = 0; l < h.g.depth; l++) {
            for (int kind = 0; kind < 2; kind++) {
                ggml_backend_tensor_get(kind ? kv.v[l] : kv.k[l], got.data(), 0, got.size() * sizeof(float));
                const array & r = ref[(kind ? "v_" : "k_") + std::to_string(l)];
                cmp c;
                for (int i = 0; i < P; i++) cmp_add(c, got.data() + (size_t)i * hd, r.f() + (size_t)valid_prefix[i] * hd, hd);
                if (l == 0 && kind == 0) ck_first = c;
                if (l == h.g.depth - 1) (kind ? cv_last : ck_last) = c;
            }
        }
        cmp_print("k layer 0", ck_first); cmp_print("k last layer", ck_last); cmp_print("v last layer", cv_last);
        cmp c1; cmp_add(c1, v_t1.data(), ref["v_t_at_t1"].f(), v_t1.size()); cmp_print("v_t at t=1", c1);
        cmp ca; cmp_add(ca, actions.data(), ref["actions"].f(), actions.size()); cmp_print("actions", ca);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(kv.buf); ggml_free(kv.ctx);
    ggml_backend_buffer_free(m.buf); ggml_free(m.ctx);
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}
