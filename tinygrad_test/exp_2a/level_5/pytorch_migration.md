# PyTorch → Tinygrad Migration Guide

This guide covers common patterns and their tinygrad equivalents.

---

## Tensor Creation

| PyTorch | Tinygrad |
|---------|----------|
| `torch.tensor([1,2,3])` | `Tensor([1,2,3])` |
| `torch.zeros(2, 3)` | `Tensor.zeros(2, 3)` |
| `torch.ones(2, 3)` | `Tensor.ones(2, 3)` |
| `torch.rand(2, 3)` | `Tensor.rand(2, 3)` |
| `torch.randn(2, 3)` | `Tensor.randn(2, 3)` |
| `torch.arange(0, 10)` | `Tensor.arange(0, 10)` |
| `torch.eye(3)` | `Tensor.eye(3)` |
| `torch.full((2,3), 5.0)` | `Tensor.full((2,3), 5.0)` |
| `torch.from_numpy(arr)` | `Tensor(arr)` |

```python
import numpy as np
from tinygrad import Tensor

arr = np.array([1.0, 2.0, 3.0])
t = Tensor(arr)          # wraps numpy array
t2 = Tensor([1, 2, 3])  # from Python list
```

---

## Device Placement

```python
# PyTorch
x = torch.tensor([1.0]).to("cuda")
x = torch.tensor([1.0]).cuda()

# Tinygrad — specify at creation or use .to()
x = Tensor([1.0], device="GPU")
x = Tensor([1.0]).to("GPU")
x = x.to("CPU")   # move to CPU
x.to_("GPU")       # in-place move
```

Tinygrad device strings: `"CPU"`, `"GPU"`, `"METAL"`, `"CUDA"`, `"CLANG"`.

---

## Training Mode

```python
# PyTorch
model.train()
# ... training loop ...
model.eval()

# Tinygrad — use context manager
from tinygrad import Tensor

with Tensor.train():
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    opt.step()

# Inference (outside context) — Tensor.training == False automatically
out = model(x)
```

---

## Gradient Management

```python
# PyTorch
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Tinygrad
opt.zero_grad()
loss.backward()
opt.step()
```

```python
# PyTorch no_grad context
with torch.no_grad():
    out = model(x)

# Tinygrad
with Tensor.no_grad():
    out = model(x)
```

---

## Optimizers

```python
from tinygrad.nn.optim import SGD, Adam, AdamW

params = nn.state.get_parameters(model)

# SGD
opt = SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam
opt = Adam(params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)

# AdamW
opt = AdamW(params, lr=1e-3, weight_decay=0.01)
```

| PyTorch | Tinygrad |
|---------|----------|
| `torch.optim.SGD(params, lr, momentum, weight_decay)` | `SGD(params, lr, momentum, weight_decay)` |
| `torch.optim.Adam(params, lr, betas, eps)` | `Adam(params, lr, b1, b2, eps)` |
| `torch.optim.AdamW(params, lr, weight_decay)` | `AdamW(params, lr, weight_decay)` |

---

## Neural Network Layers

### Linear

```python
# PyTorch
import torch.nn as torch_nn
lin = torch_nn.Linear(in_features=128, out_features=64)

# Tinygrad
from tinygrad import nn
lin = nn.Linear(128, 64)

# Usage
x = Tensor.rand(8, 128)
out = lin(x)   # shape: (8, 64)
```

### BatchNorm

```python
# PyTorch
bn = torch_nn.BatchNorm2d(num_features=32)

# Tinygrad — BatchNorm handles 1D, 2D, 3D inputs based on shape
bn = nn.BatchNorm(32)

# Usage — expects (N, C) or (N, C, *spatial)
x = Tensor.rand(8, 32, 4, 4)
out = bn(x)
```

### LayerNorm

```python
# PyTorch
ln = torch_nn.LayerNorm(normalized_shape=64)

# Tinygrad
ln = nn.LayerNorm(64)

x = Tensor.rand(8, 10, 64)
out = ln(x)
```

### Embedding

```python
# PyTorch
emb = torch_nn.Embedding(num_embeddings=1000, embedding_dim=128)

# Tinygrad
emb = nn.Embedding(1000, 128)

idx = Tensor([[1, 2, 3], [4, 5, 6]])  # integer tensor
out = emb(idx)   # shape: (2, 3, 128)
```

### Conv2d

```python
# PyTorch
conv = torch_nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Tinygrad
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

x = Tensor.rand(1, 3, 32, 32)
out = conv(x)   # shape: (1, 16, 32, 32)
```

---

## Loss Functions

```python
# PyTorch                              # Tinygrad
# MSE
loss = torch_nn.functional.mse_loss(out, target)
loss = ((out - target) ** 2).mean()

# Binary cross-entropy
loss = torch_nn.functional.binary_cross_entropy(out, target)
loss = out.binary_crossentropy(target)

# Binary cross-entropy with logits
loss = torch_nn.functional.binary_cross_entropy_with_logits(logits, target)
loss = logits.binary_crossentropy_logits(target)

# Cross-entropy (sparse labels)
loss = torch_nn.functional.cross_entropy(logits, labels)
loss = logits.sparse_categorical_crossentropy(labels)
```

---

## Model Save / Load

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))

# Tinygrad — uses safetensors format
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

state_dict = get_state_dict(model)
safe_save(state_dict, "model.safetensors")

# Load back
state_dict = safe_load("model.safetensors")
load_state_dict(model, state_dict)
```

---

## Getting Model Parameters

```python
# PyTorch
params = list(model.parameters())

# Tinygrad
from tinygrad.nn.state import get_parameters
params = get_parameters(model)
```

---

## Tensor Operations Cheat Sheet

| PyTorch | Tinygrad |
|---------|----------|
| `x.reshape(2, -1)` | `x.reshape(2, -1)` |
| `x.view(2, -1)` | `x.reshape(2, -1)` or `x.view(2, -1)` |
| `x.transpose(0, 1)` | `x.transpose(0, 1)` |
| `x.permute(2, 0, 1)` | `x.permute(2, 0, 1)` |
| `torch.cat([a, b], dim=0)` | `a.cat(b, dim=0)` or `Tensor.cat(a, b, dim=0)` |
| `torch.stack([a, b])` | `Tensor.stack([a, b])` |
| `x.squeeze(0)` | `x.squeeze(0)` |
| `x.unsqueeze(0)` | `x.unsqueeze(0)` |
| `x.mean(dim=-1)` | `x.mean(axis=-1)` |
| `x.sum(dim=0)` | `x.sum(axis=0)` |
| `x.max(dim=1)` | `x.max(axis=1)` |
| `x.softmax(dim=-1)` | `x.softmax(axis=-1)` |
| `x.detach()` | `x.detach()` |
| `x.numpy()` | `x.numpy()` |
| `x.item()` | `x.item()` |
| `x.dtype` | `x.dtype` |
| `x.shape` | `x.shape` |
| `x.device` | `x.device` |
| `x.float()` | `x.float()` |
| `x.half()` | `x.half()` |

---

## Full Training Loop Example

```python
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save


class SimpleMLP:
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.fc1(x).relu()
        return self.fc2(x)


model = SimpleMLP()
opt = Adam(get_parameters(model), lr=1e-3)

# Training
with Tensor.train():
    for step in range(100):
        x = Tensor.rand(32, 784)
        y = Tensor.randint(32, low=0, high=10)

        logits = model(x)
        loss = logits.sparse_categorical_crossentropy(y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step}, loss={loss.numpy():.4f}")

# Inference
out = model(Tensor.rand(8, 784))   # Tensor.training is False here
preds = out.argmax(axis=-1).numpy()
print("Predictions:", preds)

# Save
safe_save(get_state_dict(model), "simple_mlp.safetensors")
```
