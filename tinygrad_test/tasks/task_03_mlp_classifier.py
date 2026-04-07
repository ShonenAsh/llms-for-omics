"""
TASK 03: Multi-Layer Perceptron (MLP) Classifier

Build a 3-layer MLP for multi-class classification using tinygrad's nn module.

Architecture:
  Linear(in_features → hidden) → ReLU → Dropout(p)
  → Linear(hidden → hidden) → ReLU
  → Linear(hidden → num_classes)
"""
# Import only necessary packages.


class MLP:
    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout_p: float = 0.1):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass


def cross_entropy(y_pred: "Tensor", y_true: "Tensor") -> "Tensor":
    """Softmax cross-entropy loss. y_pred: logits (N, C), y_true: integer labels (N,)."""
    pass


def accuracy(logits: "Tensor", y_true: "Tensor") -> float:
    """Fraction of correctly classified samples, returned as a Python float."""
    pass


def train_step(model: MLP, optim, X_batch: "Tensor", Y_batch: "Tensor") -> "Tensor":
    """
    One training step: zero gradients, forward, loss, backward, optimizer step.
    Returns the loss tensor.
    """
    pass
