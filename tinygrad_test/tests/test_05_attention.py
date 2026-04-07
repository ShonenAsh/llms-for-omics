import numpy as np
import pytest
from tinygrad import Tensor
from task_05_attention import scaled_dot_product_attention, causal_attention, MultiHeadAttention


rng = np.random.default_rng(7)


def test_5a_sdpa_output_shape():
    B, T, S, d_k, d_v = 2, 5, 7, 8, 16
    Q = Tensor(rng.standard_normal((B, T, d_k)).astype(np.float32))
    K = Tensor(rng.standard_normal((B, S, d_k)).astype(np.float32))
    V = Tensor(rng.standard_normal((B, S, d_v)).astype(np.float32))
    ctx, w = scaled_dot_product_attention(Q, K, V)
    assert ctx.shape == (B, T, d_v)
    assert w.shape   == (B, T, S)


def test_5a_weights_sum_to_one():
    B, T, S, dk = 2, 4, 6, 8
    Q = Tensor(rng.standard_normal((B, T, dk)).astype(np.float32))
    K = Tensor(rng.standard_normal((B, S, dk)).astype(np.float32))
    V = Tensor(rng.standard_normal((B, S, dk)).astype(np.float32))
    _, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w.numpy().sum(axis=-1), 1.0, atol=1e-5)


def test_5b_causal_mask():
    B, T, d = 2, 6, 8
    Q = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    K = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    V = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    _, w = causal_attention(Q, K, V)
    w_np = w.numpy()
    for b in range(B):
        for i in range(T):
            for j in range(i + 1, T):
                assert w_np[b, i, j] < 1e-5, f"future ({i},{j}) not masked: {w_np[b,i,j]}"


def test_5b_causal_output_shape():
    B, T, d = 2, 5, 8
    Q = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    K = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    V = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    ctx, _ = causal_attention(Q, K, V)
    assert ctx.shape == (B, T, d)


def test_5c_mha_output_shape():
    B, T, d = 2, 5, 32
    mha = MultiHeadAttention(d, 4)
    x   = Tensor(rng.standard_normal((B, T, d)).astype(np.float32))
    out = mha(x)
    assert out.shape == (B, T, d)


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
