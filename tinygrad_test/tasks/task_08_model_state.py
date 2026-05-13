"""
TASK 08: Model Serialization

Use tinygrad's nn.state utilities to inspect, save, load, and selectively
freeze model weights.

Type hints for each function are provided as strings, replace them with real types.
"""
# Import only necessary packages here.


# Provided — do not modify.
class TinyModel:
    def __init__(self):
        from tinygrad import nn
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def __call__(self, x: "Tensor") -> "Tensor":
        return self.fc2(self.fc1(x).relu())


def get_state(model: TinyModel) -> dict:
    """Return the model's state dict (keys like 'fc1.weight', 'fc1.bias', ...)."""
    pass


def save_and_load(model: TinyModel, path: str) -> dict:
    """Save model weights to path using safetensor format, then load and return them."""
    pass


def copy_weights(src: TinyModel, dst: TinyModel) -> None:
    """Copy all parameters from src into dst so that dst(x) == src(x)."""
    pass


def freeze_fc1(model: TinyModel) -> list["Tensor"]:
    """
    Set requires_grad=False on all fc1 parameters and requires_grad=True on
    all fc2 parameters. Return the list of parameters with requires_grad=True.
    """
    pass
