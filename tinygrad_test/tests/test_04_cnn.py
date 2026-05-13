import numpy as np
import pytest
from tinygrad import Tensor
_import_error = None
try:
    from task_04_cnn import SmallCNN, count_params, forward_backward
except Exception as _e:
    _import_error = _e
    SmallCNN = count_params = forward_backward = None

def _check_import():
    if _import_error is not None:
        pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")


def test_4a_output_shape():
    _check_import()
    rng = np.random.default_rng(1)
    model = SmallCNN(1, 28, 28, 10)
    Tensor.training = False
    x = Tensor(rng.random((4, 1, 28, 28)).astype(np.float32))
    out = model(x)
    assert out.shape == (4, 10)


def test_4b_param_count():
    _check_import()
    model = SmallCNN(1, 28, 28, 10)
    n = count_params(model)
    assert n > 1000, f"param count suspiciously low: {n}"


def test_4c_forward_backward():
    _check_import()
    rng = np.random.default_rng(1)
    B, C, H, W, NC = 4, 1, 28, 28, 10
    model = SmallCNN(C, H, W, NC)
    X = Tensor(rng.random((B, C, H, W)).astype(np.float32))
    Y = Tensor(rng.integers(0, NC, B))
    logits, loss = forward_backward(model, X, Y)
    assert logits.shape == (B, NC)
    assert loss.shape == ()
    assert loss.numpy() > 0


def test_4c_rgb_input():
    _check_import()
    rng = np.random.default_rng(2)
    model = SmallCNN(3, 32, 32, 5)
    Tensor.training = False
    x = Tensor(rng.random((2, 3, 32, 32)).astype(np.float32))
    out = model(x)
    assert out.shape == (2, 5)


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
