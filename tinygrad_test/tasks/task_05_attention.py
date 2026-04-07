"""
TASK 05: Scaled Dot-Product Attention

Implement the core transformer attention mechanism from scratch using only
tinygrad Tensor operations.

Reference: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
  — "Attention is All You Need", Vaswani et al. 2017
"""
# Import only necessary packages.


def scaled_dot_product_attention(Q: "Tensor", K: "Tensor", V: "Tensor") -> tuple["Tensor", "Tensor"]:
    """
    Q: (B, T, d_k)
    K: (B, S, d_k)
    V: (B, S, d_v)
    Returns: (context, attn_weights) with shapes (B, T, d_v) and (B, T, S).
    """
    pass


def causal_attention(Q: "Tensor", K: "Tensor", V: "Tensor") -> tuple["Tensor", "Tensor"]:
    """
    Self-attention where position i may only attend to positions j <= i.
    Q, K, V: (B, T, d_k)
    Returns: (context, attn_weights) both shape (B, T, T).
    """
    pass


class MultiHeadAttention:
    """
    Split d_model into num_heads heads of size d_k = d_model // num_heads.
    Uses separate Linear projections for Q, K, V and the output.
    """
    def __init__(self, d_model: int, num_heads: int):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass
