"""
TASK 06: JIT-Compiled Training Loop

In production tinygrad, training steps are compiled with TinyJit to eliminate
Python overhead. Implement a JIT-wrapped training step, a cosine LR schedule,
and a full training loop that applies it.

Type hints for each function are provided as strings, replace them with real types.
"""
# Import only necessary packages here.


# Provided — do not modify.
class TwoLayerNet:
    def __init__(self, in_features: int, hidden: int, out_features: int):
        from tinygrad import nn
        self.l1 = nn.Linear(in_features, hidden)
        self.l2 = nn.Linear(hidden, out_features)

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.l2(self.l1(x).relu())


def make_train_step(model: TwoLayerNet, optim) -> "TinyJit":
    """
    Return a TinyJit-compiled function step(X, Y) -> loss that performs:
    forward → sparse_categorical_crossentropy loss → backward → optim step.
    """
    pass


def cosine_lr(t: int, lr_max: float, lr_min: float, T_max: int) -> float:
    """
    Cosine annealing schedule.
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
    """
    pass


def training_loop(
    model: TwoLayerNet,
    optim,
    X_np: "np.ndarray",
    Y_np: "np.ndarray",
    epochs: int = 100,
    batch_size: int = 64,
    lr_max: float = 3e-3,
    lr_min: float = 1e-4,
) -> list[float]:
    """
    Train model for `epochs` epochs, sampling random batches of size `batch_size`.
    Apply cosine LR annealing across all steps. Update optim.lr each step with:
      optim.lr.assign(Tensor([new_lr], requires_grad=False))
    Return a list of mean per-epoch losses (Python floats).
    """
    pass
