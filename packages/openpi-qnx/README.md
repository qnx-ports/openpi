# openpi-qnx

Serve a Physical Intelligence **pi0** policy on QNX 8.0 with ONNX Runtime, using openpi's own websocket protocol, so the
stock `openpi_client.WebsocketClientPolicy` works unchanged. No torch or jax on the target.

```
openpi-qnx-serve --model-dir <bundle> [--port 8000] [--profile pi0_libero] [--threads N]
python3 -m openpi_qnx.check_golden <bundle>      # compare with openpi's Policy outputs shipped in the bundle
curl http://<host>:8000/healthz                   # -> OK
```

## Model bundle (made on Linux by `export_pi0_onnx.py`)

| File | What |
|---|---|
| `prefix_encoder.onnx` | SigLIP + PaliGemma over 3 cameras + prompt -> per-layer K/V cache |
| `denoise_step.onnx` | action expert, one flow-matching step (run `steps` times) |
| `meta.json` | shapes, `openpi_config` (server profile), `action_horizon`, time-embedding constants, `embed_dtype` |
| `embed_tokens.npy` | PaliGemma token embedding table, memory-mapped (f32, or raw bf16 bits) |
| `paligemma_tokenizer.model` | sentencepiece model (`gs://big_vision/paligemma_tokenizer.model`) |
| `norm_stats.json` | the checkpoint's `assets/<asset_id>/norm_stats.json` |
| `golden.msgpack` | optional: observations, noise and openpi Policy actions for `check_golden` |

## Profiles (openpi TrainConfig reproduced)

`pi0_libero` (delta mask 6,-1), `pi0_droid` (action_horizon 10), `pi0_aloha` (adapt_to_pi, delta 6,-1,6,-1),
`pi0_aloha_sim` (no delta, default prompt "Transfer cube").

Ported from openpi 215abfb (Apache-2.0): transforms in `transforms.py`, wire protocol in `serve.py`.
