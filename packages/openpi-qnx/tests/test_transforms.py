"""Model-free unit tests (run in the APK check()). Expected values are from openpi 215abfb's own functions."""
import numpy as np
import pytest

from openpi_qnx import transforms as T


def test_make_bool_mask():
    assert T.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert T.make_bool_mask(6, -1) == (True,) * 6 + (False,)


def test_pad_to_dim():
    x = T.pad_to_dim(np.ones((2, 3)), 5)
    assert x.shape == (2, 5) and x[:, 3:].sum() == 0
    assert T.pad_to_dim(np.ones(7), 4).shape == (7,)


@pytest.mark.parametrize("quantile", [False, True])
def test_normalize_roundtrip(quantile):
    r = np.random.default_rng(0)
    s = {"state": T.NormStats(mean=r.normal(size=8).astype(np.float32), std=r.uniform(0.5, 2, 8).astype(np.float32),
                              q01=-r.uniform(1, 2, 8).astype(np.float32), q99=r.uniform(1, 2, 8).astype(np.float32))}
    x = r.normal(size=8).astype(np.float32)
    n = T.normalize({"state": x.copy()}, s, quantile)["state"]
    back = T.unnormalize({"state": T.pad_to_dim(n, 32)}, s, quantile)["state"]
    np.testing.assert_allclose(back[:8], x, rtol=1e-5, atol=1e-5)
    assert back.shape == (32,)


def test_absolute_actions_mask():
    state = np.arange(32, dtype=np.float32)
    out = T.absolute_actions({"state": state, "actions": np.zeros((3, 32), np.float32)}, T.make_bool_mask(6, -1))
    np.testing.assert_array_equal(out["actions"][0, :7], [0, 1, 2, 3, 4, 5, 0])
    np.testing.assert_array_equal(out["actions"][:, 7:], 0)


def test_aloha_gripper_values():
    # openpi aloha_policy._gripper_to_angular / _gripper_from_angular at fixed points
    np.testing.assert_allclose(T._gripper_to_angular(np.array([0.5])), [-0.14005592], rtol=1e-6)
    np.testing.assert_allclose(T._gripper_from_angular(np.array([0.5])), [0.79008664], rtol=1e-6)


def test_libero_inputs_layout():
    obs = {"observation/state": np.zeros(8), "observation/image": np.zeros((3, 224, 224), np.float32),
           "observation/wrist_image": np.zeros((224, 224, 3), np.uint8), "prompt": b"hi"}
    d = T.libero_inputs(obs)
    assert d["image"]["base_0_rgb"].shape == (224, 224, 3) and d["image"]["base_0_rgb"].dtype == np.uint8
    assert d["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
    assert d["prompt"] == "hi"


def test_resize_with_pad_shape():
    out = T.resize_images({"a": np.zeros((480, 640, 3), np.uint8)})
    assert out["a"].shape == (224, 224, 3)
