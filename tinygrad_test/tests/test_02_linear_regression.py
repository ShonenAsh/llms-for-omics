import numpy as np
import pytest
from tinygrad import Tensor
from task_02_linear_regression import init_params, predict, mse_loss, sgd_step, train


def test_2a_init_params_shapes():
    w, b = init_params(5)
    assert w.shape == (5,)
    assert b.shape == ()
    assert w.requires_grad
    assert b.requires_grad


def test_2b_predict_shape():
    rng = np.random.default_rng(0)
    x = Tensor(rng.random((10, 5)).astype(np.float32))
    w, b = init_params(5)
    y_hat = predict(x, w, b)
    assert y_hat.shape == (10,)


def test_2c_mse_loss():
    y_hat = Tensor(np.array([1.0, 2.0, 3.0]))
    y     = Tensor(np.array([1.5, 2.5, 3.5]))
    loss  = mse_loss(y_hat, y)
    assert loss.shape == ()
    assert np.allclose(loss.numpy(), 0.25, atol=1e-6)


def test_2e_train_recovers_weights():
    rng = np.random.default_rng(42)
    N, D = 200, 3
    true_w = np.array([1.5, -2.0, 0.5], dtype=np.float32)
    true_b = np.float32(0.3)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ true_w + true_b + rng.standard_normal(N).astype(np.float32) * 0.05

    w, b = train(X, Y, lr=0.05, steps=500)
    assert np.allclose(w.numpy(), true_w, atol=0.15), f"w: {w.numpy()} vs {true_w}"
    assert np.allclose(b.numpy(), true_b, atol=0.15), f"b: {b.numpy()} vs {true_b}"


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
