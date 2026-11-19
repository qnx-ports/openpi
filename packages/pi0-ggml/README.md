# pi0-ggml

pi0 inference in C++ on [ggml](https://github.com/ggml-org/ggml), with no Python, torch or jax at run time. It was written to
run pi0 on a GPU under QNX 8.0 through ggml's Vulkan backend, and builds wherever ggml does.

- **prefix:** SigLIP over the valid camera images, then Gemma 2B over [image tokens, language tokens]. The per-layer K/V
  stays in a GPU-resident cache.
- **denoise:** the Gemma 300M action expert over [state token, 50 action tokens], attending to that cache. It runs for
  10 Euler steps, the same sampling as `PI0Pytorch.sample_actions`.
- Padded prefix tokens (cameras with `img_mask = 0`, language padding) are dropped rather than masked. openpi masks them
  out of every valid attention row, and their RoPE positions equal the compacted index, so valid outputs are unchanged.

The language embeddings are an input (`lang_emb`), so the 257k-row vocabulary table is not part of the model. See
`packages/openpi-qnx` for how a server tokenises the prompt and gathers embedding rows from a memory-mapped table.

## Build

Against an installed ggml. For QNX, use `qnx-ports/ggml` branch `qnx-0.14.0` (tag `qnx-0.14.0-r0` once merged),
configured with `-DGGML_VULKAN=ON`.

```
cmake -S packages/pi0-ggml -B build -DCMAKE_PREFIX_PATH=<ggml install prefix>
cmake --build build
```

Alternatively, copy this directory to `ggml/examples/pi0` and add `add_subdirectory(pi0)` to `ggml/examples/CMakeLists.txt`.

## Run

```
pi0-run (-m pi0.gguf | --random tiny|full [--type q8_0|f16|f32] [--depth N] [--vision-depth N])
        [--ref DIR] [--device gpu|cpu] [-t THREADS] [--runs N] [--cams N] [--lang N]
```

- `-m` loads a GGUF written by `scripts/qnx/export_pi0_gguf.py`.
- `--random` builds random weights of the given size in memory, which is enough for timing.
- `--ref DIR` compares the K/V cache, the first image embedding, `v_t` at t = 1 and the final actions with the PyTorch
  reference that the exporter writes next to the GGUF.

## Status (2026-11)

All results so far use **random weights**. Real checkpoints and accuracy with real weights are still to do.

Agreement with PyTorch (rel = max|diff| / max|ref|):

| Model | Backend | Type | Actions |
|---|---|---|---|
| tiny | CPU | f32 | 1.1e-5 |
| real widths, 2+2 layers | CPU | f32 | 1.1e-4 |
| real widths, 2+2 layers | Vulkan (NVK) | q8_0 | 1.2e-2 |

Speed with full-size pi0 Q8_0 (3.0 GB of weights) on an RTX 4070 Laptop GPU, QNX 8.0, Mesa 25.1.9 NVK, without
cooperative matrix:

| Prefix | Total per inference |
|---|---|
| 2 cameras + 20 tokens | 1.67 s |
| 3 cameras + 48 tokens | 2.23 s |

The first inference in a process adds 3-4 s for Vulkan pipeline creation.

Known issue: ggml-vulkan Q8_0 gives NaN for the tiny model's 64-wide matrices. Real pi0 widths (all >= 1024) are fine.
