# pi0 on QNX 8.0

This fork, [qnx-ports/openpi](https://github.com/qnx-ports/openpi), is based on upstream openpi 215abfb. It adds what is
needed to serve Physical Intelligence's pi0 policy on QNX 8.0 targets, with no torch or jax on the target.
openpi itself is unchanged; everything QNX-related is in three directories:

| Path | What |
|---|---|
| [`scripts/qnx/`](../scripts/qnx/README.md) | Linux-side exporters, from openpi's PyTorch pi0 to ONNX graphs (for ONNX Runtime) and to GGUF (for ggml), plus checking tools |
| [`packages/openpi-qnx/`](../packages/openpi-qnx/README.md) | The policy server: openpi's websocket protocol, the transforms for pi0_libero, pi0_droid, pi0_aloha and pi0_aloha_sim, and the tokenizer, with ONNX Runtime inference. The stock `openpi_client` works against it |
| [`packages/pi0-ggml/`](../packages/pi0-ggml/README.md) | pi0 in C++ on ggml, for GPU inference through ggml's Vulkan backend |

The QNX packages (APKBUILDs) are in [qnx-ports/aports](https://github.com/qnx-ports/aports).
The QNX changes to ggml are in [qnx-ports/ggml](https://github.com/qnx-ports/ggml) (branch `qnx-0.14.0`).

Status, as of 2026-11:
- The ONNX Runtime server matches openpi's own `Policy` within 2e-6 on x86_64 and aarch64 QNX, with the tiny model.
- pi0-ggml runs full-size pi0 on an RTX 4070 Laptop GPU under QNX in 1.7-2.2 s per inference.
- Both have used random weights so far.
