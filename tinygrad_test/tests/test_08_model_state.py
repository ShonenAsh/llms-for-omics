import pathlib, tempfile
import numpy as np
import pytest
from tinygrad import Tensor
from task_08_model_state import TinyModel, get_state, save_and_load, copy_weights, freeze_fc1


EXPECTED_KEYS = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def test_8a_state_dict_keys():
    model = TinyModel()
    sd = get_state(model)
    assert set(sd.keys()) == EXPECTED_KEYS


def test_8a_state_dict_shapes():
    model = TinyModel()
    sd = get_state(model)
    assert sd["fc1.weight"].shape == (8, 4)
    assert sd["fc1.bias"].shape   == (8,)
    assert sd["fc2.weight"].shape == (2, 8)
    assert sd["fc2.bias"].shape   == (2,)


def test_8b_save_and_load_roundtrip():
    model = TinyModel()
    sd    = get_state(model)
    with tempfile.TemporaryDirectory() as tmp:
        p = str(pathlib.Path(tmp) / "model.safetensors")
        loaded = save_and_load(model, p)
        assert set(loaded.keys()) == EXPECTED_KEYS
        for k in EXPECTED_KEYS:
            assert np.allclose(loaded[k].numpy(), sd[k].numpy()), f"mismatch: {k}"


def test_8c_copy_weights_equal_outputs():
    rng = np.random.default_rng(9)
    src, dst = TinyModel(), TinyModel()
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    copy_weights(src, dst)
    Tensor.training = False
    assert np.allclose(src(x).numpy(), dst(x).numpy(), atol=1e-6)


def test_8d_freeze_fc1_leaves_fc2_trainable():
    model = TinyModel()
    trainable = freeze_fc1(model)
    sd = get_state(model)
    trainable_keys = {k for k, v in sd.items() if v.requires_grad is True}
    frozen_keys    = {k for k, v in sd.items() if v.requires_grad is False}
    assert trainable_keys == {"fc2.weight", "fc2.bias"}
    assert frozen_keys    == {"fc1.weight", "fc1.bias"}
    assert len(trainable) == 2


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
