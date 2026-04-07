"""
TASK 04: Convolutional Neural Network (CNN)

Build a small CNN for image classification with the following architecture:

  Block 1: Conv2d(in_ch → 32, 3×3, padding=1) → BatchNorm(32) → ReLU → MaxPool(2×2)
  Block 2: Conv2d(32 → 64, 3×3, padding=1)    → BatchNorm(64) → ReLU → MaxPool(2×2)
  Classifier: flatten → Linear(64 * (H//4) * (W//4) → 128) → ReLU → Linear(128 → num_classes)
"""
# Import only necessary packages.


class SmallCNN:
    def __init__(self, in_ch: int, img_h: int, img_w: int, num_classes: int):
        pass

    def __call__(self, x: "Tensor") -> "Tensor":
        pass


def count_params(model: SmallCNN) -> int:
    """Return the total number of scalar parameters in the model."""
    pass


def forward_backward(model: SmallCNN, X: "Tensor", Y: "Tensor") -> tuple["Tensor", "Tensor"]:
    """
    Run a forward pass, compute sparse_categorical_crossentropy loss, and
    call backward(). Return (logits, loss) as realized Tensors.
    """
    pass
