import numpy as np
import pytest
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from task_06_training_loop import TwoLayerNet, make_train_step, cosine_lr, training_loop


def test_6b_cosine_lr_endpoints():
    lr = cosine_lr(0, lr_max=1e-2, lr_min=1e-4, T_max=100)
    assert abs(lr - 1e-2) < 1e-7, f"t=0: {lr}"
    lr_end = cosine_lr(100, lr_max=1e-2, lr_min=1e-4, T_max=100)
    assert abs(lr_end - 1e-4) < 1e-6, f"t=T_max: {lr_end}"


def test_6b_cosine_lr_midpoint():
    lr_mid = cosine_lr(50, lr_max=1e-2, lr_min=1e-4, T_max=100)
    expected = (1e-2 + 1e-4) / 2
    assert abs(lr_mid - expected) < 1e-6, f"t=50: {lr_mid}"


def test_6b_cosine_lr_monotone():
    lrs = [cosine_lr(t, 1e-2, 1e-4, 100) for t in range(101)]
    assert all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1)), "LR should be non-increasing"


def test_6c_training_returns_correct_length():
    rng = np.random.default_rng(3)
    N, D, C = 200, 8, 4
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = rng.integers(0, C, N)
    model = TwoLayerNet(D, 32, C)
    optim = Adam(nn.state.get_parameters(model), lr=3e-3)
    losses = training_loop(model, optim, X, Y, epochs=10, batch_size=32)
    assert len(losses) == 10


def test_6c_training_decreases_loss():
    rng = np.random.default_rng(3)
    N, D, C = 400, 8, 4
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = rng.integers(0, C, N)
    model = TwoLayerNet(D, 32, C)
    optim = Adam(nn.state.get_parameters(model), lr=3e-3)
    losses = training_loop(model, optim, X, Y, epochs=50, batch_size=64)
    assert losses[-1] < losses[0], f"loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
