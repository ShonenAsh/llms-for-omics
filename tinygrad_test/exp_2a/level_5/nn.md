## Neural Network classes

#### nn.BatchNorm(sz: 'int', eps=1e-05, affine=True, track_running_stats=True, momentum=0.1)

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


#### nn.Conv1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding: 'int | str' = 0, dilation=1, groups=1, bias=True) -> Conv2d

Applies a 1D convolution over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

```python
conv = nn.Conv1d(1, 1, 3)
t = Tensor.rand(1, 1, 4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = conv(t)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.Conv2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding: 'int | tuple[int, ...] | str' = 0, dilation=1, groups=1, bias=True)

Applies a 2D convolution over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

```python
conv = nn.Conv2d(1, 1, 3)
t = Tensor.rand(1, 1, 4, 4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = conv(t)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.ConvTranspose1d(in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True) -> ConvTranspose2d

Applies a 1D transposed convolution operator over an input signal composed of several input planes.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

```python
conv = nn.ConvTranspose1d(1, 1, 3)
t = Tensor.rand(1, 1, 4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = conv(t)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.ConvTranspose2d(in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, ...]', stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True)

Applies a 2D transposed convolution operator over an input image.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d

```python
conv = nn.ConvTranspose2d(1, 1, 3)
t = Tensor.rand(1, 1, 4, 4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = conv(t)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.Linear(in_features: 'int', out_features: 'int', bias=True)

Applies a linear transformation to the incoming data.

See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

```python
lin = nn.Linear(3, 4)
t = Tensor.rand(2, 3)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = lin(t)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.GroupNorm(num_groups: 'int', num_channels: 'int', eps=1e-05, affine=True)

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


#### nn.InstanceNorm(num_features: 'int', eps: 'float' = 1e-05, affine: 'bool' = True)

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


#### nn.LayerNorm(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)

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


#### nn.LayerNorm2d(normalized_shape: 'int | tuple[int, ...]', eps: 'float' = 1e-05, elementwise_affine: 'bool' = True)

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


#### nn.RMSNorm(dim: 'int', eps=1e-06, elementwise_affine=True)

Applies Root Mean Square Normalization to input.

- Paper: https://arxiv.org/abs/1910.07467

```python
norm = nn.RMSNorm(4)
t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(norm(t).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.Embedding(vocab_size: 'int', embed_size: 'int')

A simple lookup table that stores embeddings of a fixed dictionary and size.

See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding

```python
emb = nn.Embedding(10, 3)
print(emb(Tensor([1, 2, 3, 1])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### nn.LSTMCell(input_size: 'int', hidden_size: 'int', bias: 'bool' = True)

A long short-term memory (LSTM) cell.

Args:
  input_size: The number of expected features in the input `x`
  hidden_size: The number of features in the hidden state `h`
  bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`


## Optimizers

#### nn.optim.SGD(params: list[tinygrad.tensor.Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False, fused=<ContextVar>)

Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

`classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.


#### nn.optim.LARS(params: list[tinygrad.tensor.Tensor], lr=0.001, momentum=0.9, weight_decay=0.0001, ns_steps=0, ns_coefficients=None, nesterov=False, classic=True, pre_wd=True, tcoef=0.001, fused=<ContextVar>)

Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

- Paper: https://arxiv.org/abs/1708.03888v3


#### nn.optim.AdamW(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-08, weight_decay=0.01, fused=<ContextVar>)

AdamW optimizer with optional weight decay.

- Paper: https://arxiv.org/abs/1711.05101v3


#### nn.optim.Adam(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-08, fused=<ContextVar>)

Adam optimizer.

- Paper: https://arxiv.org/abs/1412.6980


#### nn.optim.LAMB(params: list[tinygrad.tensor.Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-06, weight_decay=0.0, adam=False, fused=<ContextVar>)

LAMB optimizer with optional weight decay.

- Paper: https://arxiv.org/abs/1904.00962


## Load/Save

#### nn.state.safe_load(fn: tinygrad.tensor.Tensor | str | pathlib.Path) -> dict[str, tinygrad.tensor.Tensor]

Loads a .safetensor file, returning the `state_dict`.


#### nn.state.safe_save(tensors: dict[str, tinygrad.tensor.Tensor], fn: str, metadata: dict[str, Any] | None = None)

Saves a `state_dict` to disk in a .safetensor file with optional metadata.


#### nn.state.get_state_dict(obj, prefix: str = '', tensor_type=<class 'tinygrad.tensor.Tensor'>) -> dict[str, tinygrad.tensor.Tensor]

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


#### nn.state.get_parameters(obj) -> list[tinygrad.tensor.Tensor]

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


#### nn.state.load_state_dict(model, state_dict: dict[str, tinygrad.tensor.Tensor], strict=True, verbose=True, consume=False, realize=True) -> list[tinygrad.tensor.Tensor]

Loads a `state_dict` into a model. Return the loaded Tensors.


#### nn.state.tar_extract(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### nn.state.torch_load(t: tinygrad.tensor.Tensor) -> dict[str, tinygrad.tensor.Tensor]


#### nn.state.gguf_load(tensor: tinygrad.tensor.Tensor) -> tuple[dict, dict[str, tinygrad.tensor.Tensor]]

Loads a .gguf file, returning the `kv_data` and `state_dict`.

