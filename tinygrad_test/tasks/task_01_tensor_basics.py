"""
TASK 01: Tensor Basics

Practice creating and manipulating tinygrad Tensors.
Implement each function below. Do not change signatures.

Type hints for each function are provided as strings, replace them with real types.
"""
# Import only necessary packages here.



def ones_times_five() -> "Tensor":
    """Return a (3, 4) tensor where every element is 5.0."""
    pass


def matmul(A: "np.ndarray", B: "np.ndarray") -> "Tensor":
    """Wrap A and B as tinygrad Tensors and return their matrix product."""
    pass


def reduce_last_dim(x: "Tensor") -> tuple["Tensor", "Tensor"]:
    """
    Given x of shape (B, T, C), return a tuple (mean, maximum) where both
    are computed over the last dimension, each with shape (B, T).
    """
    pass


def manual_relu(x: "Tensor") -> "Tensor":
    """ReLU without calling x.relu()."""
    pass


def manual_sigmoid(x: "Tensor") -> "Tensor":
    """Sigmoid without calling x.sigmoid()."""
    pass


def outer_sum(x: "Tensor", y: "Tensor") -> "Tensor":
    """
    Given x of shape (8,) and y of shape (8,), return a (8, 8) tensor
    where result[i, j] = x[i] + y[j], using broadcasting.
    """
    pass
