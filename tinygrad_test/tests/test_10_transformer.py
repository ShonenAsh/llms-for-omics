import numpy as np
import pytest
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.optim import Adam
from task_10_transformer import CausalMHA, FFN, TransformerBlock, MiniLM


rng = np.random.default_rng(42)
VOCAB, D, HEADS, LAYERS, SEQ, B = 32, 64, 4, 2, 16, 2


def make_model():
    return MiniLM(VOCAB, D, HEADS, LAYERS, SEQ)


def test_10a_causal_mha_shape():
    mha = CausalMHA(D, HEADS)
    x   = Tensor(rng.standard_normal((B, SEQ, D)).astype(np.float32))
    out = mha(x)
    assert out.shape == (B, SEQ, D)


def test_10b_ffn_shape():
    ffn = FFN(D)
    x   = Tensor(rng.standard_normal((B, SEQ, D)).astype(np.float32))
    assert ffn(x).shape == (B, SEQ, D)


def test_10c_block_shape():
    block = TransformerBlock(D, HEADS)
    x     = Tensor(rng.standard_normal((B, SEQ, D)).astype(np.float32))
    assert block(x).shape == (B, SEQ, D)


def test_10c_minilm_logits_shape():
    model  = make_model()
    ids    = Tensor(rng.integers(0, VOCAB, (B, SEQ)))
    Tensor.training = False
    logits = model(ids)
    assert logits.shape == (B, SEQ, VOCAB)


def test_10d_generation_length():
    model   = make_model()
    Tensor.training = False
    prompt  = Tensor(rng.integers(0, VOCAB, (1, 4)))
    out     = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (1, 10)


def test_10e_training_decreases_loss():
    model = make_model()
    optim = Adam(nn.state.get_parameters(model), lr=3e-3)

    def step(x, y):
        Tensor.training = True
        optim.zero_grad()
        loss = model(x).reshape(-1, VOCAB).sparse_categorical_crossentropy(y.reshape(-1))
        loss.backward()
        optim.step()
        return loss

    jit_step = TinyJit(step)
    losses = []
    for _ in range(60):
        seq = Tensor(rng.integers(0, VOCAB, (B, SEQ)))
        tgt = Tensor(rng.integers(0, VOCAB, (B, SEQ)))
        losses.append(jit_step(seq, tgt).numpy().item())

    assert losses[-1] < losses[0], f"loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
