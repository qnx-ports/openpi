#!/usr/bin/env python3
"""Reference outputs from openpi's own Policy for a bundle, and a check of openpi_qnx against them.

Builds the transform chain openpi.policies.policy_config.create_trained_policy builds for the bundle's openpi
config, wraps the bundle's PyTorch weights (torch_model.pt, tiny export) in openpi.policies.policy.Policy, and runs
openpi's make_*_example() observations through it with fixed noise. Writes golden.msgpack (observations, noise,
expected actions) into the bundle so the same comparison can run on the target, then runs openpi_qnx.QnxPolicy
(ONNX Runtime) on this host against the same goldens.

  PYTHONPATH=src:packages/openpi-qnx python packages/openpi-qnx/tests/make_golden.py <bundle_dir> [n_obs]   (from the repo root)
"""
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "qnx"))
import export_pi0_onnx as E  # noqa: E402
from openpi import transforms as T  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: E402
from openpi.policies import aloha_policy, droid_policy, libero_policy  # noqa: E402
from openpi.policies.policy import Policy  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402

PI0 = _model.ModelType.PI0


def openpi_chain(config: str):
    """(data inputs, data outputs, default_prompt, example fn) exactly as openpi 215abfb training/config.py."""
    if config == "pi0_libero":  # LeRobotLiberoDataConfig(extra_delta_transform=True)
        m = T.make_bool_mask(6, -1)
        return ([libero_policy.LiberoInputs(model_type=PI0), T.DeltaActions(m)],
                [T.AbsoluteActions(m), libero_policy.LiberoOutputs()], None, libero_policy.make_libero_example)
    if config == "pi0_droid":
        return ([droid_policy.DroidInputs(model_type=PI0)], [droid_policy.DroidOutputs()], None,
                droid_policy.make_droid_example)
    if config == "pi0_aloha":  # LeRobotAlohaDataConfig defaults: adapt_to_pi, use_delta_joint_actions
        m = T.make_bool_mask(6, -1, 6, -1)
        return ([aloha_policy.AlohaInputs(adapt_to_pi=True), T.DeltaActions(m)],
                [T.AbsoluteActions(m), aloha_policy.AlohaOutputs(adapt_to_pi=True)], None,
                aloha_policy.make_aloha_example)
    if config == "pi0_aloha_sim":
        return ([aloha_policy.AlohaInputs(adapt_to_pi=True)], [aloha_policy.AlohaOutputs(adapt_to_pi=True)],
                "Transfer cube", aloha_policy.make_aloha_example)
    raise ValueError(config)


def main():
    bundle = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    meta = json.load(open(f"{bundle}/meta.json"))
    if meta["size"] != "tiny":
        raise SystemExit("golden generation reuses the tiny export path; for full size pass real weights instead")
    E.gp.PaliGemmaForConditionalGeneration = E.tiny_vision(E.gp.PaliGemmaForConditionalGeneration)
    cfg = types.SimpleNamespace(pi05=False, paligemma_variant="dummy", action_expert_variant="dummy",
                                dtype="float32", action_dim=meta["action_dim"], action_horizon=meta["action_horizon"],
                                pytorch_compile_mode=None)
    model = E.PI0Pytorch(cfg).eval()
    model.load_state_dict(torch.load(f"{bundle}/torch_model.pt"))

    data_in, data_out, default_prompt, example = openpi_chain(meta["openpi_config"])
    norm = _normalize.deserialize_json(open(f"{bundle}/norm_stats.json").read())
    policy = Policy(model,
                    transforms=[T.InjectDefaultPrompt(default_prompt), *data_in, T.Normalize(norm, use_quantiles=False),
                                T.InjectDefaultPrompt(None), T.ResizeImages(224, 224),
                                T.TokenizePrompt(PaligemmaTokenizer(meta["lang_len"])),
                                T.PadStatesAndActions(meta["action_dim"])],
                    output_transforms=[T.Unnormalize(norm, use_quantiles=False), *data_out],
                    is_pytorch=True, pytorch_device="cpu")

    rng = np.random.default_rng(1234)
    np.random.seed(1234)  # make_*_example uses np.random
    prompts = ["pick up the red block", "place the cup on the table", "do something"]
    cases = []
    for i in range(n):
        obs = example()
        if default_prompt is None or i % 2 == 0:
            obs["prompt"] = prompts[i % len(prompts)]
        else:
            obs.pop("prompt", None)  # exercise InjectDefaultPrompt
        noise = rng.standard_normal((meta["action_horizon"], meta["action_dim"])).astype(np.float32)
        with torch.no_grad():
            out = policy.infer(dict(obs), noise=noise)
        cases.append({"obs": obs, "noise": noise, "actions": np.asarray(out["actions"], np.float32)})
        print(f"case {i}: prompt={obs.get('prompt', '<default>')!r} actions {out['actions'].shape}")
    with open(f"{bundle}/golden.msgpack", "wb") as f:
        f.write(msgpack_numpy.packb({"openpi_config": meta["openpi_config"], "cases": cases}))

    from openpi_qnx.policy import QnxPolicy
    q = QnxPolicy(bundle)
    worst = max(float(np.abs(q.infer(dict(c["obs"]), noise=c["noise"])["actions"] - c["actions"]).max()) for c in cases)
    print(f"{meta['openpi_config']}: openpi Policy (torch) vs openpi_qnx (ORT {__import__('onnxruntime').__version__})"
          f" max |actions diff| = {worst:.3g}  -> {'PASS' if worst < 1e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
