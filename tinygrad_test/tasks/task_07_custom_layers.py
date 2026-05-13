"""
TASK 07: Custom Layers

Implement neural-network building blocks not provided by tinygrad.nn,
using raw Tensor operations.

Type hints for each function are provided as strings, replace them with real types.
"""

# Import only necessary packages here.


class LayerNorm:
    """
    Normalize over the last dimension.
    Formula: (x - mean) / sqrt(var + eps) * weight + bias
    weight and bias are learnable parameters of shape (normalized_shape,).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass


class Embedding:
    """
    Learned embedding table of shape (vocab_size, d_model).
    Maps integer token ids (B, T) to dense vectors (B, T, d_model).
    """
    def __init__(self, vocab_size: int, d_model: int):
        pass

    def __call__(self, idx: "Tensor") -> "Tensor":
        pass


class ResidualBlock:
    """
    Pre-norm residual block:
      LayerNorm → Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)
    with a residual connection around the whole block.
    """
    def __init__(self, d_model: int):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass
