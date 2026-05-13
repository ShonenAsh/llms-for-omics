import numpy as np
import pytest
from tinygrad import Tensor
_import_error = None
try:
    from task_09_custom_losses import focal_loss, dice_loss, contrastive_loss
except Exception as _e:
    _import_error = _e
    focal_loss = dice_loss = contrastive_loss = None

def _check_import():
    if _import_error is not None:
        pytest.fail(f"Could not import task module: {type(_import_error).__name__}: {_import_error}")


rng = np.random.default_rng(11)


def test_9a_focal_loss_shape():
    _check_import()
    logits  = Tensor(rng.standard_normal(50).astype(np.float32))
    targets = Tensor(rng.integers(0, 2, 50).astype(np.float32))
    loss = focal_loss(logits, targets)
    assert loss.shape == ()


def test_9a_focal_loss_positive():
    _check_import()
    logits  = Tensor(rng.standard_normal(100).astype(np.float32))
    targets = Tensor(rng.integers(0, 2, 100).astype(np.float32))
    assert focal_loss(logits, targets).numpy() > 0


def test_9a_focal_le_bce():
    _check_import()
    logits  = Tensor(rng.standard_normal(100).astype(np.float32))
    targets = Tensor(rng.integers(0, 2, 100).astype(np.float32))
    fl  = focal_loss(logits, targets).numpy()
    bce = (-(targets * logits.sigmoid().log() + (1 - targets) * (1 - logits.sigmoid()).log())).mean().numpy()
    assert fl <= bce + 1e-4, f"focal ({fl:.4f}) > bce ({bce:.4f})"


def test_9b_dice_loss_range():
    _check_import()
    N, H, W = 4, 16, 16
    pred   = Tensor(rng.random((N, H, W)).astype(np.float32))
    target = Tensor((rng.random((N, H, W)) > 0.5).astype(np.float32))
    dl = dice_loss(pred, target).numpy()
    assert 0 <= dl <= 1, f"dice loss out of [0,1]: {dl}"


def test_9b_dice_loss_perfect():
    _check_import()
    target = Tensor((rng.random((4, 8, 8)) > 0.5).astype(np.float32))
    dl = dice_loss(target, target).numpy()
    assert dl < 0.05, f"perfect dice loss: {dl}"


def test_9c_contrastive_loss_non_negative():
    _check_import()
    N, D = 8, 16
    emb1   = Tensor(rng.standard_normal((N, D)).astype(np.float32))
    emb2   = Tensor(rng.standard_normal((N, D)).astype(np.float32))
    labels = Tensor(rng.integers(0, 2, N).astype(np.float32))
    assert contrastive_loss(emb1, emb2, labels).numpy() >= 0


def test_9c_contrastive_identical_similar_zero():
    _check_import()
    N, D = 8, 16
    same   = Tensor(rng.standard_normal((N, D)).astype(np.float32))
    labels = Tensor(np.zeros(N, dtype=np.float32))
    cl = contrastive_loss(same, same, labels).numpy()
    assert cl < 1e-5, f"same embeddings, similar label → expected ≈0, got {cl}"


def test_no_unused_imports(unused_imports):
    assert not unused_imports, f"Unnecessary imports: {unused_imports}"
