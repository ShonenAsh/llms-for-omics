"""
TASK 10: Transformer Block & Mini Language Model

Assemble a complete Transformer decoder and a small language model.

Transformer block (pre-norm style):
  x → LayerNorm → CausalMultiHeadAttention → + residual
    → LayerNorm → FFN (Linear → GELU → Linear) → + residual

Language model:
  token ids → Embedding + positional Embedding
            → N × TransformerBlock
            → LayerNorm
            → Linear(d_model, vocab_size)
"""
# Import only necessary packages.


# Provided — do not modify.
class LayerNorm:
    def __init__(self, d: int, eps: float = 1e-5):
        from tinygrad import Tensor
        self.weight = Tensor.ones(d, requires_grad=True)
        self.bias   = Tensor.zeros(d, requires_grad=True)
        self.eps = eps

    def __call__(self, x: "Tensor") -> "Tensor":
        mean = x.mean(axis=-1, keepdim=True)
        var  = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.weight + self.bias


class CausalMHA:
    """
    Multi-head self-attention with causal masking.
    Uses nn.Linear projections for Q, K, V and the output.
    d_k = d_model // num_heads.
    """
    def __init__(self, d_model: int, num_heads: int):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass


class FFN:
    """Feed-forward sub-layer: Linear(d → 4d) → GELU → Linear(4d → d)."""
    def __init__(self, d_model: int):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass


class TransformerBlock:
    """Pre-norm transformer decoder block with causal self-attention and FFN."""
    def __init__(self, d_model: int, num_heads: int):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass


class MiniLM:
    """Small autoregressive language model."""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, max_seq_len: int):
        pass

    def __call__(self, idx: "Tensor") -> "Tensor":
        """idx: (B, T) integer token ids. Returns logits (B, T, vocab_size)."""
        pass

    def generate(self, idx: "Tensor", max_new_tokens: int) -> "Tensor":
        """Greedily append max_new_tokens tokens to the prompt idx."""
        pass
