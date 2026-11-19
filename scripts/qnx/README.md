# scripts/qnx: export and check pi0 for QNX

These are Linux-side tools that turn openpi's PyTorch pi0 (`src/openpi/models_pytorch`) into models for QNX 8.0 targets.
Run them from the repository root, with openpi's own environment (`PYTHONPATH=src`).

| Script | What it does |
|---|---|
| `export_pi0_onnx.py` | Exports pi0 as two ONNX graphs, `prefix_encoder` and `denoise_step`, plus `meta.json`, the embedding table, the tokenizer and the norm stats: the bundle that `packages/openpi-qnx` serves with ONNX Runtime. Checks ORT against torch on export. |
| `export_pi0_gguf.py` | Converts pi0 to GGUF (f32, f16 or q8_0) for `packages/pi0-ggml`, plus a raw reference set for `pi0-run --ref`. Imports `export_pi0_onnx.py`. |
| `validate_ort.py` | Runs an ONNX bundle with ONNX Runtime and compares it with the exported reference. This also runs on QNX. |
| `bench_ort.py` | Times the ONNX graphs with ONNX Runtime (median of N), for latency estimates. |
| `dump_raw.py` | Writes the reference inputs and outputs as raw binaries for the C++ harness. |
| `infer.cpp` | C++ harness (ONNX Runtime C++ API) that runs the two graphs and the 10-step sampling loop. |

Sizes: `--size tiny` uses Gemma's "dummy" variants and a shrunken SigLIP, for fast tests. `--size bench` uses real pi0
widths with only a few layers, for timing. `--size full` is the real model; pass `--weights <dir>` for a converted checkpoint.

```
PYTHONPATH=src python scripts/qnx/export_pi0_onnx.py --size tiny --out out/onnx-tiny [--save-torch]
PYTHONPATH=src python scripts/qnx/export_pi0_gguf.py --size tiny --type f32 --out out/gguf-tiny
```

`export_pi0_onnx.py` wraps openpi's `denoise_step` with int casts before the CumSum/ReduceSum on boolean masks. ONNX
export needs them, and they give the same output as upstream (max diff 0.0). Making upstream's code export cleanly would
remove the wrapper, which makes it a candidate to send upstream.
