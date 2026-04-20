## Neural Network classes

#### BatchNorm(sz: 'int', eps=1e-05, affine=True, track_running_stats=True, momentum=0.1)

Applies Batch Normalization over a 2D or 3D input.

- Paper: https://arxiv.org/abs/1502.03167v3

See: `Tensor.batchnorm`

```python
norm = nn.BatchNorm(3)
t = Tensor.rand(2, 3, 4, 4)
print(t.mean().item(), t.std().item())
```


#### Conv1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding: 'int | str' = 0, dilation=1, groups=1, bias=True) -> Conv2d

Applies a 1D convolution over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

```python
conv = nn.Conv1d(1, 1, 3)
t = Tensor.rand(1, 1, 4)
print(t.numpy())
```


#### Conv2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding: 'int | tuple[int, ...] | str' = 0, dilation=1, groups=1, bias=True)

Applies a 2D convolution over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

```python
conv = nn.Conv2d(1, 1, 3)
t = Tensor.rand(1, 1, 4, 4)
print(t.numpy())
```


#### ConvTranspose1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True) -> ConvTranspose2d

Applies a 1D transposed convolution operator over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

```python
conv = nn.ConvTranspose1d(1, 1, 3)
t = Tensor.rand(1, 1, 4)
print(t.numpy())
```


#### ConvTranspose2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True)

Applies a 2D transposed convolution operator over an input image.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d

```python
conv = nn.ConvTranspose2d(1, 1, 3)
t = Tensor.rand(1, 1, 4, 4)
print(t.numpy())
```


#### Linear(in_features: 'int', out_features: 'int', bias=True)

Applies a linear transformation to the incoming data.

See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

```python
lin = nn.Linear(3, 4)
t = Tensor.rand(2, 3)
print(t.numpy())
```


#### GroupNorm(num_groups: 'int', num_channels: 'int', eps=1e-05, affine=True)

Applies Group Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1803.08494v3

```python
norm = nn.GroupNorm(2, 12)
t = Tensor.rand(2, 12, 4, 4) * 2 + 1
print(t.mean().item(), t.std().item())
```


#### InstanceNorm(num_features: 'int', eps: 'float' = 1e-05, affine: 'bool' = True)

Applies Instance Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1607.08022v3

```python
norm = nn.InstanceNorm(3)
t = Tensor.rand(2, 3, 4, 4) * 2 + 1
print(t.mean().item(), t.std().item())
```


#### LayerNorm(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)

Applies Layer Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1607.06450v1

```python
norm = nn.LayerNorm(3)
t = Tensor.rand(2, 5, 3) * 2 + 1
print(t.mean().item(), t.std().item())
```


#### LayerNorm2d(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)

Applies Layer Normalization over a mini-batch of 2D inputs.

See: `LayerNorm`

```python
norm = nn.LayerNorm2d(3)
t = Tensor.rand(2, 3, 4, 4) * 2 + 1
print(t.mean().item(), t.std().item())
```


#### RMSNorm(dim: 'int', eps=1e-06, elementwise_affine=True)

Applies Root Mean Square Normalization to input.

- Paper: https://arxiv.org/abs/1910.07467

```python
norm = nn.RMSNorm(4)
t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
print(t.numpy())
```


#### Embedding(vocab_size: 'int', embed_size: 'int')

A simple lookup table that stores embeddings of a fixed dictionary and size.

See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding

```python
emb = nn.Embedding(10, 3)
print(emb(Tensor([1, 2, 3, 1])).numpy())
```


#### LSTMCell(input_size: 'int', hidden_size: 'int', bias: 'bool' = True)

A long short-term memory (LSTM) cell.

Args:
  input_size: The number of expected features in the input `x`
  hidden_size: The number of features in the hidden state `h`
  bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`


## Optimizers

#### SGD(params: list[tinygrad.tensor.Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False, fused=<ContextVar>)

Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

`classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.


#### LARS(params: list[tinygrad.tensor.Tensor], lr=0.001, momentum=0.9, weight_decay=0.0001, ns_steps=0, ns_coefficients=None, nesterov=False, classic=True, pre_wd=True, tcoef=0.001, fused=<ContextVar>)

Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

- Paper: https://arxiv.org/abs/1708.03888v3


#### AdamW(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-08, weight_decay=0.01, fused=<ContextVar>)

AdamW optimizer with optional weight decay.

- Paper: https://arxiv.org/abs/1711.05101v3


#### Adam(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-08, fused=<ContextVar>)

Adam optimizer.

- Paper: https://arxiv.org/abs/1412.6980


#### LAMB(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-06, weight_decay=0.0, adam=False, fused=<ContextVar>)

LAMB optimizer with optional weight decay.

- Paper: https://arxiv.org/abs/1904.00962


## Load/Save

#### safe_load(fn: tinygrad.tensor.Tensor | str | pathlib._local.Path) -> dict[str, tinygrad.tensor.Tensor]

Loads a .safetensor file, returning the `state_dict`.


#### safe_save(tensors: dict[str, tinygrad.tensor.Tensor], fn: str, metadata: dict[str, typing.Any] | None = None)

Saves a `state_dict` to disk in a .safetensor file with optional metadata.


#### get_state_dict(obj, prefix: str = '', tensor_type=<class 'tinygrad.tensor.Tensor'>) -> dict[str, tinygrad.tensor.Tensor]

Returns a `state_dict` of the object, with optional prefix.

```python
class Net:
  def __init__(self):
    self.l1 = nn.Linear(4, 5)
    self.l2 = nn.Linear(5, 6)

net = Net()
print(nn.state.get_state_dict(net).keys())
```


#### get_parameters(obj) -> list[tinygrad.tensor.Tensor]

```python
class Net:
  def __init__(self):
    self.l1 = nn.Linear(4, 5)
    self.l2 = nn.Linear(5, 6)

net = Net()
print(len(nn.state.get_parameters(net)))
```


#### load_state_dict(model, state_dict: dict[str, tinygrad.tensor.Tensor], strict=True, verbose=True, consume=False, realize=True) -> list[tinygrad.tensor.Tensor]

Loads a `state_dict` into a model. Return the loaded Tensors.


#### tar_extract(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### torch_load(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### gguf_load(tensor: tinygrad.tensor.Tensor) -> tuple[dict, dict[str, tinygrad.tensor.Tensor]]

Loads a .gguf file, returning the `kv_data` and `state_dict`.

