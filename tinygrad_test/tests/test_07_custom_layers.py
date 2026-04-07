import numpy as np
import pytest
from tinygrad import Tensor
from task_07_custom_layers import LayerNorm, Embedding, ResidualBlock


rng = np.random.default_rng(5)


def test_7a_layernorm_shape():
    x  = Tensor(rng.standard_normal((2, 6, 16)).astype(np.float32))
    ln = LayerNorm(16)
    assert ln(x).shape == (2, 6, 16)


def test_7a_layernorm_normalizes():
    x  = Tensor(rng.standard_normal((4, 8, 32)).astype(np.float32))
    ln = LayerNorm(32)
    out = ln(x).numpy()
    # Each feature vector should have mean≈0, std≈1 (before affine)
    # With default weight=1 and bias=0, mean should be ≈0
    for b in range(4):
        for t in range(8):
            assert abs(out[b, t].mean()) < 0.2, f"mean not near 0 at ({b},{t})"


def test_7a_layernorm_learnable_params():
    ln = LayerNorm(16)
    assert ln.weight.requires_grad
    assert ln.bias.requires_grad
    assert ln.weight.shape == (16,)
    assert ln.bias.shape   == (16,)


def test_7b_embedding_shape():
    emb = Embedding(50, 32)
    ids = Tensor(rng.integers(0, 50, (2, 6)))
    out = emb(ids)
    assert out.shape == (2, 6, 32)


def test_7b_embedding_consistent():
    emb  = Embedding(50, 32)
    ids1 = Tensor([[3, 3]])
    out  = emb(ids1).numpy()
    assert np.allclose(out[0, 0], out[0, 1]), "same token → same embedding"


def test_7c_residual_block_shape():
    d = 64
    rb = ResidualBlock(d)
    x  = Tensor(rng.standard_normal((2, 6, d)).astype(np.float32))
    assert rb(x).shape == (2, 6, d)


def test_7c_residual_connection_present():
    """When fc weights are zeroed out the block should output the input unchanged."""
    d  = 8
    rb = ResidualBlock(d)
    # Zero all linear weights so FFN contributes nothing
    for layer in [rb.fc1, rb.fc2]:
        layer.weight.assign(Tensor.zeros_like(layer.weight))
        layer.bias.assign(Tensor.zeros_like(layer.bias))
    x   = Tensor(rng.standard_normal((2, 3, d)).astype(np.float32))
    out = rb(x).numpy()
    # With zeroed FFN, output ≈ x (residual path dominates)
    assert not np.allclose(out, 0), "residual connection seems missing (all zeros output)"


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
