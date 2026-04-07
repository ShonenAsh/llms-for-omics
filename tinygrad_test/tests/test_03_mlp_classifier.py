import numpy as np
import pytest
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from task_03_mlp_classifier import MLP, cross_entropy, accuracy, train_step


def test_3a_mlp_output_shape():
    model  = MLP(16, 64, 5)
    x      = Tensor(np.random.randn(8, 16).astype(np.float32))
    Tensor.training = False
    out    = model(x)
    assert out.shape == (8, 5)


def test_3b_cross_entropy_positive():
    model  = MLP(8, 32, 4)
    x      = Tensor(np.random.randn(16, 8).astype(np.float32))
    y      = Tensor(np.random.randint(0, 4, 16))
    Tensor.training = False
    logits = model(x)
    loss   = cross_entropy(logits, y)
    assert loss.shape == ()
    assert loss.numpy() > 0


def test_3c_accuracy_range():
    logits = Tensor(np.random.randn(100, 5).astype(np.float32))
    y      = Tensor(np.random.randint(0, 5, 100))
    acc    = accuracy(logits, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_3d_training_reduces_loss():
    rng   = np.random.default_rng(0)
    N, D, C = 500, 16, 5
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = rng.integers(0, C, N)
    model = MLP(D, 64, C)
    optim = Adam(nn.state.get_parameters(model), lr=3e-3)

    first_losses, last_losses = [], []
    for step in range(300):
        idx = rng.integers(0, N, 64)
        Xb, Yb = Tensor(X[idx]), Tensor(Y[idx])
        loss = train_step(model, optim, Xb, Yb)
        if step < 10:
            first_losses.append(loss.numpy().item())
        if step >= 290:
            last_losses.append(loss.numpy().item())

    assert np.mean(last_losses) < np.mean(first_losses), "loss did not decrease"

    Tensor.training = False
    acc = accuracy(model(Tensor(X)), Tensor(Y))
    assert acc > 0.4, f"accuracy too low: {acc:.3f}"


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
