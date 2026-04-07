import numpy as np
import pytest
from tinygrad import Tensor
from task_01_tensor_basics import ones_times_five, matmul, reduce_last_dim, manual_relu, manual_sigmoid, outer_sum


def test_1a_ones_times_five():
    t = ones_times_five()
    assert t.shape == (3, 4)
    assert np.allclose(t.numpy(), 5.0)


def test_1b_matmul():
    rng = np.random.default_rng(0)
    A = rng.random((4, 3)).astype(np.float32)
    B = rng.random((3, 5)).astype(np.float32)
    out = matmul(A, B)
    assert out.shape == (4, 5)
    assert np.allclose(out.numpy(), A @ B, atol=1e-5)


def test_1c_reduce_last_dim():
    rng = np.random.default_rng(0)
    x_np = rng.random((2, 6, 8)).astype(np.float32)
    x = Tensor(x_np)
    mean, maximum = reduce_last_dim(x)
    assert mean.shape == (2, 6)
    assert maximum.shape == (2, 6)
    assert np.allclose(mean.numpy(), x_np.mean(axis=-1), atol=1e-5)
    assert np.allclose(maximum.numpy(), x_np.max(axis=-1), atol=1e-5)


def test_1d_manual_relu():
    v = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert np.allclose(manual_relu(v).numpy(), np.maximum(v.numpy(), 0))


def test_1d_manual_sigmoid():
    v = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert np.allclose(manual_sigmoid(v).numpy(), 1 / (1 + np.exp(-v.numpy())), atol=1e-6)


def test_1e_outer_sum():
    x = Tensor(np.arange(8, dtype=np.float32))
    y = Tensor(np.arange(8, dtype=np.float32))
    out = outer_sum(x, y)
    assert out.shape == (8, 8)
    assert np.allclose(out.numpy(), x.numpy().reshape(8, 1) + y.numpy())


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
