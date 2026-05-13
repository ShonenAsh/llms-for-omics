"""
TASK 09: Custom Loss Functions

Implement three loss functions using tinygrad Tensor primitives.

Type hints for each function are provided as strings, replace them with real types.
"""
# Import only necessary packages here.


def focal_loss(logits: "Tensor", targets: "Tensor", alpha: float = 0.25, gamma: float = 2.0) -> "Tensor":
    """
    Binary focal loss for class-imbalanced problems.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logits) for the true class:
      p_t = sigmoid(logits)       when targets == 1
      p_t = 1 - sigmoid(logits)  when targets == 0

    logits : (N,) raw scores
    targets: (N,) binary labels in {0, 1}
    Returns: scalar mean loss.
    """
    pass


def dice_loss(pred: "Tensor", target: "Tensor", smooth: float = 1.0) -> "Tensor":
    """
    Soft Dice loss for binary segmentation.

    Dice = (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
    Loss = 1 - Dice, averaged over the batch.

    pred  : (N, H, W) predicted probabilities in [0, 1]
    target: (N, H, W) binary ground-truth masks
    Returns: scalar mean loss.
    """
    pass


def contrastive_loss(emb1: "Tensor", emb2: "Tensor", labels: "Tensor", margin: float = 1.0) -> "Tensor":
    """
    Contrastive loss for metric learning (Hadsell et al. 2006).

    L = (1 - y) * 0.5 * D^2  +  y * 0.5 * max(0, margin - D)^2

    where D is the L2 distance between embeddings and y=1 means dissimilar.

    emb1, emb2: (N, D) embedding tensors
    labels    : (N,)   0 = similar pair, 1 = dissimilar pair
    Returns: scalar mean loss.
    """
    pass
