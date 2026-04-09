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

```
0.5357538461685181 0.3014977276325226
```

```python
t = norm(t)
print(t.mean().item(), t.std().item())
```

```
0.5357511639595032 0.3014962077140808
```


#### Conv1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding: 'int | str' = 0, dilation=1, groups=1, bias=True) -> Conv2d

Applies a 1D convolution over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

```python
conv = nn.Conv1d(1, 1, 3)
t = Tensor.rand(1, 1, 4)
print(t.numpy())
```

```
[[[0.4271 0.2283 0.0146 0.1937]]]
```

```python
t = conv(t)
print(t.numpy())
```

```
[[[0.4648 0.3716]]]
```


#### Conv2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding: 'int | tuple[int, ...] | str' = 0, dilation=1, groups=1, bias=True)

Applies a 2D convolution over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

```python
conv = nn.Conv2d(1, 1, 3)
t = Tensor.rand(1, 1, 4, 4)
print(t.numpy())
```

```
[[[[0.0826 0.8581 0.5957 0.7321]
   [0.4712 0.7846 0.8044 0.532 ]
   [0.449  0.3396 0.8071 0.6423]
   [0.7522 0.1629 0.0704 0.3497]]]]
```

```python
t = conv(t)
print(t.numpy())
```

```
[[[[0.361  0.2647]
   [0.2894 0.2034]]]]
```


#### ConvTranspose1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True) -> ConvTranspose2d

Applies a 1D transposed convolution operator over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

```python
conv = nn.ConvTranspose1d(1, 1, 3)
t = Tensor.rand(1, 1, 4)
print(t.numpy())
```

```
[[[0.3342 0.7717 0.6324 0.0343]]]
```

```python
t = conv(t)
print(t.numpy())
```

```
[[[0.083  0.0926 0.2242 0.3328 0.2281 0.1171]]]
```


#### ConvTranspose2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True)

Applies a 2D transposed convolution operator over an input image.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d

```python
conv = nn.ConvTranspose2d(1, 1, 3)
t = Tensor.rand(1, 1, 4, 4)
print(t.numpy())
```

```
[[[[0.118  0.952  0.7915 0.6112]
   [0.441  0.8924 0.7023 0.4594]
   [0.4927 0.7309 0.4189 0.0639]
   [0.4936 0.9978 0.3558 0.4155]]]]
```

```python
t = conv(t)
print(t.numpy())
```

```
[[[[ 0.2938  0.0653  0.0549 -0.0466  0.1544  0.2112]
   [ 0.2181  0.1667  0.1937  0.383   0.4371  0.4077]
   [ 0.2522  0.3817  0.5654  0.6085  0.4839  0.3948]
   [ 0.3053  0.3278  0.5042  0.3363  0.3656  0.2315]
   [ 0.4457  0.6215  0.6772  0.669   0.4289  0.4351]
   [ 0.3995  0.5336  0.4607  0.3562  0.3485  0.2948]]]]
```


#### Linear(in_features: 'int', out_features: 'int', bias=True)

Applies a linear transformation to the incoming data.

See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

```python
lin = nn.Linear(3, 4)
t = Tensor.rand(2, 3)
print(t.numpy())
```

```
[[0.5698 0.4578 0.0563]
 [0.9243 0.8321 0.5801]]
```

```python
t = lin(t)
print(t.numpy())
```

```
[[ 0.111  -0.733  -0.0297 -0.0338]
 [ 0.0782 -0.8291 -0.1169  0.0595]]
```


#### GroupNorm(num_groups: 'int', num_channels: 'int', eps=1e-05, affine=True)

Applies Group Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1803.08494v3

```python
norm = nn.GroupNorm(2, 12)
t = Tensor.rand(2, 12, 4, 4) * 2 + 1
print(t.mean().item(), t.std().item())
```

```
2.047156810760498 0.5902590751647949
```

```python
t = norm(t)
print(t.mean().item(), t.std().item())
```

```
-1.0406900941006825e-07 1.0012903213500977
```


#### InstanceNorm(num_features: 'int', eps: 'float' = 1e-05, affine: 'bool' = True)

Applies Instance Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1607.08022v3

```python
norm = nn.InstanceNorm(3)
t = Tensor.rand(2, 3, 4, 4) * 2 + 1
print(t.mean().item(), t.std().item())
```

```
1.9731030464172363 0.5847082138061523
```

```python
t = norm(t)
print(t.mean().item(), t.std().item())
```

```
9.302018355583641e-08 1.0052337646484375
```


#### LayerNorm(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)

Applies Layer Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1607.06450v1

```python
norm = nn.LayerNorm(3)
t = Tensor.rand(2, 5, 3) * 2 + 1
print(t.mean().item(), t.std().item())
```

```
2.104135274887085 0.5218847990036011
```

```python
t = norm(t)
print(t.mean().item(), t.std().item())
```

```
-2.3638310153728526e-07 1.0169306993484497
```


#### LayerNorm2d(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)

Applies Layer Normalization over a mini-batch of 2D inputs.

See: `LayerNorm`

```python
norm = nn.LayerNorm2d(3)
t = Tensor.rand(2, 3, 4, 4) * 2 + 1
print(t.mean().item(), t.std().item())
```

```
2.0451512336730957 0.5732497572898865
```

```python
t = norm(t)
print(t.mean().item(), t.std().item())
```

```
-2.3422410322382348e-07 1.005179524421692
```


#### RMSNorm(dim: 'int', eps=1e-06, elementwise_affine=True)

Applies Root Mean Square Normalization to input.

- Paper: https://arxiv.org/abs/1910.07467

```python
norm = nn.RMSNorm(4)
t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
print(t.numpy())
```

```
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
```

```python
print(norm(t).numpy())
```

```
[[0.     0.5345 1.069  1.6036]
 [0.7127 0.8909 1.069  1.2472]
 [0.8363 0.9409 1.0454 1.15  ]]
```


#### Embedding(vocab_size: 'int', embed_size: 'int')

A simple lookup table that stores embeddings of a fixed dictionary and size.

See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding

```python
emb = nn.Embedding(10, 3)
print(emb(Tensor([1, 2, 3, 1])).numpy())
```

```
[[-0.1092  0.4184  0.1417]
 [-0.1954  0.207   0.2442]
 [ 0.4285 -0.5828  0.3009]
 [-0.1092  0.4184  0.1417]]
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

```
dict_keys(['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'])
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

```
4
```


#### load_state_dict(model, state_dict: dict[str, tinygrad.tensor.Tensor], strict=True, verbose=True, consume=False, realize=True) -> list[tinygrad.tensor.Tensor]

Loads a `state_dict` into a model. Return the loaded Tensors.


#### tar_extract(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### torch_load(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### gguf_load(tensor: tinygrad.tensor.Tensor) -> tuple[dict, dict[str, tinygrad.tensor.Tensor]]

Loads a .gguf file, returning the `kv_data` and `state_dict`.

