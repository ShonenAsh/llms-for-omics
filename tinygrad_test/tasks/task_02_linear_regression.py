"""
TASK 02: Linear Regression from Scratch

Implement linear regression using tinygrad's autograd — no optimizer classes.
"""
# Import only necessary packages.


def init_params(in_features: int) -> tuple["Tensor", "Tensor"]:
    """
    Return (w, b) where:
      - w has shape (in_features,) and requires_grad=True
      - b is a scalar (shape ()) and requires_grad=True
    Initialize w with small random values and b as zero.
    """
    pass


def predict(x: "Tensor", w: "Tensor", b: "Tensor") -> "Tensor":
    """Linear prediction: x @ w + b. Returns shape (N,)."""
    pass


def mse_loss(y_hat: "Tensor", y: "Tensor") -> "Tensor":
    """Mean squared error between predictions and targets."""
    pass


def sgd_step(w: "Tensor", b: "Tensor", lr: float) -> tuple["Tensor", "Tensor"]:
    """
    Update w and b in-place using their gradients and learning rate lr.
    Use .assign() to keep the computation graph clean. Return (w, b).
    """
    pass


def train(X: "np.ndarray", Y: "np.ndarray", lr: float = 0.1, steps: int = 200) -> tuple["Tensor", "Tensor"]:
    """
    Train linear regression on (X, Y) for `steps` gradient steps.
    Zero out gradients after each step. Return (w, b).
    """
    pass
