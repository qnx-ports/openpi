"""Input/output transforms for pi0 inference, ported to plain numpy from openpi (215abfb).

The chain is the one openpi.policies.policy_config.create_trained_policy builds:

  inputs:  InjectDefaultPrompt -> robot Inputs (+ DeltaActions, a no-op at inference) -> Normalize
           -> ResizeImages(224) -> TokenizePrompt -> PadStatesAndActions(action_dim)
  outputs: Unnormalize -> AbsoluteActions (if the profile uses delta actions) -> robot Outputs

pi0 uses z-score normalisation (openpi sets use_quantile_norm only for non-PI0 models); quantile mode is kept for
completeness. Profiles are named after the openpi TrainConfig they reproduce.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable

import numpy as np
from openpi_client import image_tools

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # openpi.models.model.IMAGE_KEYS
IMAGE_SIZE = 224


# ----------------------------------------------------------------------------- generic (openpi/transforms.py)
def pad_to_dim(x: np.ndarray, target: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    x = np.asarray(x)
    cur = x.shape[axis]
    if cur < target:
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, target - cur)
        return np.pad(x, pad, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    out: list[bool] = []
    for d in dims:
        out.extend([True] * d if d > 0 else [False] * (-d))
    return tuple(out)


@dataclasses.dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(path: str) -> dict[str, NormStats]:
    """openpi norm_stats.json: {"norm_stats": {"state": {"mean": [...], "std": [...], "q01": ..., "q99": ...}, ...}}"""
    with open(path) as f:
        raw = json.load(f)["norm_stats"]
    return {k: NormStats(**{f: (None if v.get(f) is None else np.asarray(v[f], np.float32))
                            for f in ("mean", "std", "q01", "q99")}) for k, v in raw.items()}


def normalize(data: dict, stats: dict[str, NormStats], quantile: bool) -> dict:
    for key, s in stats.items():
        if key not in data:
            continue  # openpi Normalize is non-strict: actions are absent at inference
        x = np.asarray(data[key])
        if quantile:
            q01, q99 = s.q01[..., : x.shape[-1]], s.q99[..., : x.shape[-1]]
            data[key] = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean, std = s.mean[..., : x.shape[-1]], s.std[..., : x.shape[-1]]
            data[key] = (x - mean) / (std + 1e-6)
    return data


def unnormalize(data: dict, stats: dict[str, NormStats], quantile: bool) -> dict:
    for key, s in stats.items():  # openpi Unnormalize is strict: every stats key must be present
        x = np.asarray(data[key])
        if quantile:
            q01, q99 = s.q01, s.q99
            if (dim := q01.shape[-1]) < x.shape[-1]:
                data[key] = np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], -1)
            else:
                data[key] = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            mean = pad_to_dim(s.mean, x.shape[-1], value=0.0)
            std = pad_to_dim(s.std, x.shape[-1], value=1.0)
            data[key] = x * (std + 1e-6) + mean
    return data


def absolute_actions(data: dict, mask: tuple[bool, ...] | None) -> dict:
    if "actions" not in data or mask is None:
        return data
    m = np.asarray(mask)
    dims = m.shape[-1]
    actions = np.array(data["actions"])
    actions[..., :dims] += np.expand_dims(np.where(m, data["state"][..., :dims], 0), axis=-2)
    data["actions"] = actions
    return data


def parse_image(image) -> np.ndarray:
    """libero/droid _parse_image: float [0,1] -> uint8, CHW -> HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _prompt(data: dict, out: dict) -> None:
    if "prompt" in data:
        p = data["prompt"]
        out["prompt"] = p.decode("utf-8") if isinstance(p, bytes) else p


# ----------------------------------------------------------------------------- LIBERO (libero_policy.py)
def libero_inputs(data: dict) -> dict:
    base, wrist = parse_image(data["observation/image"]), parse_image(data["observation/wrist_image"])
    out = {"state": data["observation/state"],
           "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
           "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}}
    _prompt(data, out)
    return out


def libero_outputs(data: dict) -> dict:
    return {"actions": np.asarray(data["actions"][..., :7])}


# ----------------------------------------------------------------------------- DROID (droid_policy.py, PI0 branch)
def droid_inputs(data: dict) -> dict:
    grip = np.asarray(data["observation/gripper_position"])
    if grip.ndim == 0:
        grip = grip[np.newaxis]
    state = np.concatenate([data["observation/joint_position"], grip])
    base = parse_image(data["observation/exterior_image_1_left"])
    wrist = parse_image(data["observation/wrist_image_left"])
    out = {"state": state,
           "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
           "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}}
    _prompt(data, out)
    return out


def droid_outputs(data: dict) -> dict:
    return {"actions": np.asarray(data["actions"][..., :8])}


# ----------------------------------------------------------------------------- ALOHA (aloha_policy.py)
_ALOHA_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
_FLIP = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _norm(x, lo, hi):
    return (x - lo) / (hi - lo)


def _unnorm(x, lo, hi):
    return x * (hi - lo) + lo


def _gripper_to_angular(v):
    v = _unnorm(v, 0.01844, 0.05800)
    arm, horn = 0.036, 0.022
    v = np.arcsin(np.clip((horn**2 + v**2 - arm**2) / (2 * horn * v), -1.0, 1.0))
    return _norm(v, 0.5476, 1.6296)


def _gripper_from_angular(v):
    return _norm(v + 0.5476, -0.6213, 1.4910)


def make_aloha_inputs(adapt_to_pi: bool) -> Callable[[dict], dict]:
    def aloha_inputs(data: dict) -> dict:
        state = np.asarray(data["state"])
        if adapt_to_pi:
            state = _FLIP * state
            state[[6, 13]] = _gripper_to_angular(state[[6, 13]])

        def conv(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            return np.transpose(img, (1, 2, 0))  # aloha images are always CHW

        imgs = {k: conv(v) for k, v in data["images"].items()}
        if set(imgs) - set(_ALOHA_CAMERAS):
            raise ValueError(f"Expected images to contain {_ALOHA_CAMERAS}, got {tuple(imgs)}")
        base = imgs["cam_high"]
        image, mask = {"base_0_rgb": base}, {"base_0_rgb": True}
        for dest, src in (("left_wrist_0_rgb", "cam_left_wrist"), ("right_wrist_0_rgb", "cam_right_wrist")):
            image[dest] = imgs.get(src, np.zeros_like(base))
            mask[dest] = src in imgs
        out = {"image": image, "image_mask": mask, "state": state}
        _prompt(data, out)
        return out

    return aloha_inputs


def make_aloha_outputs(adapt_to_pi: bool) -> Callable[[dict], dict]:
    def aloha_outputs(data: dict) -> dict:
        actions = np.array(data["actions"][:, :14])
        if adapt_to_pi:
            actions = _FLIP * actions
            actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
        return {"actions": actions}

    return aloha_outputs


# ----------------------------------------------------------------------------- profiles
@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    inputs: Callable[[dict], dict]
    outputs: Callable[[dict], dict]
    delta_mask: tuple[bool, ...] | None = None
    default_prompt: str | None = None
    action_horizon: int = 50
    quantile_norm: bool = False


PROFILES = {
    "pi0_libero": Profile("pi0_libero", libero_inputs, libero_outputs, delta_mask=make_bool_mask(6, -1)),
    "pi0_droid": Profile("pi0_droid", droid_inputs, droid_outputs, action_horizon=10),
    "pi0_aloha": Profile("pi0_aloha", make_aloha_inputs(True), make_aloha_outputs(True),
                         delta_mask=make_bool_mask(6, -1, 6, -1)),
    "pi0_aloha_sim": Profile("pi0_aloha_sim", make_aloha_inputs(True), make_aloha_outputs(True),
                             default_prompt="Transfer cube"),
}


def resize_images(images: dict) -> dict:
    """ResizeImages(224, 224). openpi's server uses jax.image.resize here; clients normally send 224x224 already
    (openpi_client.image_tools.resize_with_pad), in which case both are no-ops. For other sizes this uses the PIL
    version from openpi_client, which can differ slightly from jax's resize."""
    return {k: image_tools.resize_with_pad(np.asarray(v), IMAGE_SIZE, IMAGE_SIZE) for k, v in images.items()}
