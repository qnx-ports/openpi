// pi0 inference harness for QNX: prefix_encoder once, then N Euler steps of denoise_step (ONNX Runtime C++ API).
//
//   infer <model_dir> [runs] [threads]
//
// <model_dir> holds prefix_encoder.onnx, denoise_step.onnx, meta.json, and ref/ (from dump_raw.py): the same fixed
// inputs PyTorch used, plus PyTorch's outputs. The harness checks its K/V cache and final actions against them, then
// reports median latency.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Blob {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<char> data;
  size_t count() const {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    return n;
  }
};

static std::map<std::string, Blob> load_ref(const std::string& dir) {
  std::map<std::string, Blob> m;
  std::ifstream mf(dir + "/manifest.txt");
  if (!mf) { fprintf(stderr, "cannot open %s/manifest.txt\n", dir.c_str()); exit(2); }
  std::string line;
  while (std::getline(mf, line)) {
    std::istringstream ss(line);
    std::string name; Blob b; int64_t d;
    ss >> name >> b.dtype;
    while (ss >> d) b.shape.push_back(d);
    size_t esz = b.dtype == "f32" ? 4 : b.dtype == "i64" ? 8 : 1;
    b.data.resize(b.count() * esz);
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    f.read(b.data.data(), (std::streamsize)b.data.size());
    if ((size_t)f.gcount() != b.data.size()) { fprintf(stderr, "short read %s\n", name.c_str()); exit(2); }
    m[name] = std::move(b);
  }
  return m;
}

static Ort::Value view(const Ort::MemoryInfo& mi, Blob& b) {
  if (b.dtype == "f32")
    return Ort::Value::CreateTensor<float>(mi, (float*)b.data.data(), b.count(), b.shape.data(), b.shape.size());
  if (b.dtype == "bool")
    return Ort::Value::CreateTensor<bool>(mi, (bool*)b.data.data(), b.count(), b.shape.data(), b.shape.size());
  return Ort::Value::CreateTensor<int64_t>(mi, (int64_t*)b.data.data(), b.count(), b.shape.data(), b.shape.size());
}

static std::vector<std::string> names(Ort::Session& s, bool inputs) {
  Ort::AllocatorWithDefaultOptions a;
  std::vector<std::string> v;
  size_t n = inputs ? s.GetInputCount() : s.GetOutputCount();
  for (size_t i = 0; i < n; i++)
    v.emplace_back(inputs ? s.GetInputNameAllocated(i, a).get() : s.GetOutputNameAllocated(i, a).get());
  return v;
}

static std::vector<const char*> cstrs(const std::vector<std::string>& v) {
  std::vector<const char*> r;
  for (auto& s : v) r.push_back(s.c_str());
  return r;
}

// Timestep embedding exactly as openpi's create_sinusoidal_pos_embedding: computed in double, then cast to float.
// period = min_period * (max_period / min_period) ** linspace(0, 1, W/2);  emb = [sin(2*pi*t/period), cos(...)]
static void time_embedding(float t, std::vector<float>& out) {
  const double min_p = 4e-3, max_p = 4.0;  // PI0Pytorch.embed_suffix constants (also in meta.json)
  const size_t half = out.size() / 2;
  for (size_t i = 0; i < half; i++) {
    double frac = half > 1 ? (double)i / (double)(half - 1) : 0.0;
    double period = min_p * std::pow(max_p / min_p, frac);
    double x = (1.0 / period) * 2.0 * M_PI * (double)t;
    out[i] = (float)std::sin(x);
    out[half + i] = (float)std::cos(x);
  }
}

static double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <model_dir> [runs] [threads]\n", argv[0]); return 2; }
  std::string dir = argv[1];
  int runs = argc > 2 ? atoi(argv[2]) : 5;
  int threads = argc > 3 ? atoi(argv[3]) : 0;
  const int steps = 10;

  auto ref = load_ref(dir + "/ref");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pi0");
  Ort::SessionOptions so;
  if (threads > 0) so.SetIntraOpNumThreads(threads);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  auto t0 = std::chrono::steady_clock::now();
  Ort::Session sp(env, (dir + "/prefix_encoder.onnx").c_str(), so);
  Ort::Session sd(env, (dir + "/denoise_step.onnx").c_str(), so);
  printf("ORT %s, sessions loaded in %.0f ms\n", Ort::GetVersionString().c_str(), ms_since(t0));

  auto p_in = names(sp, true), p_out = names(sp, false), d_in = names(sd, true), d_out = names(sd, false);
  auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t te_shape[2] = {1, 0};
  for (size_t i = 0; i < d_in.size(); i++)
    if (d_in[i] == "time_emb") te_shape[1] = sd.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()[1];
  if (te_shape[1] <= 0) { fprintf(stderr, "denoise_step has no time_emb input\n"); return 2; }
  std::vector<float> te((size_t)te_shape[1]);

  std::vector<float> x;
  double kv_err = 0, act_err = 0;
  std::vector<double> t_pre, t_den;
  for (int r = 0; r <= runs; r++) {  // run 0 = warm-up + correctness
    std::vector<Ort::Value> pin;
    for (auto& n : p_in) pin.push_back(view(mi, ref.at(n)));
    auto tp = std::chrono::steady_clock::now();
    auto pout = sp.Run(Ort::RunOptions{nullptr}, cstrs(p_in).data(), pin.data(), pin.size(), cstrs(p_out).data(),
                       p_out.size());
    double dp = ms_since(tp);

    if (r == 0)
      for (size_t i = 1; i < p_out.size(); i++) {
        auto& rb = ref.at(p_out[i]);
        const float* got = pout[i].GetTensorData<float>();
        const float* want = (const float*)rb.data.data();
        for (size_t j = 0; j < rb.count(); j++) kv_err = std::max(kv_err, (double)std::fabs(got[j] - want[j]));
      }

    Blob& noise = ref.at("noise");
    x.assign((float*)noise.data.data(), (float*)noise.data.data() + noise.count());
    float t = 1.0f, dt = -1.0f / steps;
    auto td = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++) {
      std::vector<Ort::Value> din;
      for (auto& n : d_in) {
        if (n == "x_t") din.push_back(Ort::Value::CreateTensor<float>(mi, x.data(), x.size(), noise.shape.data(), 3));
        else if (n == "time_emb") {
          time_embedding(t, te);
          din.push_back(Ort::Value::CreateTensor<float>(mi, te.data(), te.size(), te_shape, 2));
        }
        else if (n == "state") din.push_back(view(mi, ref.at("state")));
        else {  // prefix_pad_masks and k_i/v_i: pass the prefix encoder's outputs straight through, no copy
          auto it = std::find(p_out.begin(), p_out.end(), n);
          if (it == p_out.end()) { fprintf(stderr, "unbound input %s\n", n.c_str()); return 2; }
          auto& src = pout[it - p_out.begin()];
          auto info = src.GetTensorTypeAndShapeInfo();
          auto shp = info.GetShape();
          if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
            din.push_back(Ort::Value::CreateTensor<bool>(mi, src.GetTensorMutableData<bool>(),
                                                         info.GetElementCount(), shp.data(), shp.size()));
          else
            din.push_back(Ort::Value::CreateTensor<float>(mi, src.GetTensorMutableData<float>(),
                                                          info.GetElementCount(), shp.data(), shp.size()));
        }
      }
      auto dout = sd.Run(Ort::RunOptions{nullptr}, cstrs(d_in).data(), din.data(), din.size(), cstrs(d_out).data(), 1);
      const float* v = dout[0].GetTensorData<float>();
      for (size_t j = 0; j < x.size(); j++) x[j] += dt * v[j];
      t += dt;
    }
    double dd = ms_since(td);
    if (r == 0) {
      const float* want = (const float*)ref.at("actions").data.data();
      for (size_t j = 0; j < x.size(); j++) act_err = std::max(act_err, (double)std::fabs(x[j] - want[j]));
    } else {
      t_pre.push_back(dp);
      t_den.push_back(dd);
    }
  }
  std::sort(t_pre.begin(), t_pre.end());
  std::sort(t_den.begin(), t_den.end());
  double mp = t_pre.empty() ? 0 : t_pre[t_pre.size() / 2], md = t_den.empty() ? 0 : t_den[t_den.size() / 2];
  printf("max |kv - torch| = %.3g   max |actions - torch| = %.3g\n", kv_err, act_err);
  printf("median over %d: prefix %.1f ms + %d denoise steps %.1f ms = %.1f ms\n", runs, mp, steps, md, mp + md);
  printf("actions[0][0..3] = %.5f %.5f %.5f %.5f\n", x[0], x[1], x[2], x[3]);
  bool ok = kv_err < 1e-3 && act_err < 1e-3;
  printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
