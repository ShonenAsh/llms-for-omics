## Basic

#### Tensor.shape -> tuple[sint, ...]


#### Tensor.dtype -> DType


#### Tensor.device -> str | tuple[str, ...]


#### Tensor.ndim -> <class 'int'>

Returns the number of dimensions in the tensor.

```python
t = Tensor([[1, 2], [3, 4]])
print(t.ndim)
```

```
2
```


#### Tensor.numel() -> UOp | int

Returns the total number of elements in the tensor.

```python
t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t.numel())
```

```
8
```


#### Tensor.element_size() -> int

Returns the size in bytes of an individual element in the tensor.

```python
t = Tensor([5], dtype=dtypes.int16)
print(t.element_size())
```

```
2
```


#### Tensor.nbytes() -> int

Returns the total number of bytes of all elements in the tensor.

```python
t = Tensor([8, 9], dtype=dtypes.float)
print(t.nbytes())
```

```
8
```


#### Tensor.is_floating_point() -> bool

Returns `True` if the tensor contains floating point types, i.e. is one of `dtypes.float64`, `dtypes.float32`,
`dtypes.float16`, `dtypes.bfloat16`.

```python
t = Tensor([8, 9], dtype=dtypes.float32)
print(t.is_floating_point())
```

```
True
```


#### Tensor.size(dim: 'int | None' = None) -> sint | tuple[sint, ...]

Returns the size of the tensor. If `dim` is specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.

```python
t = Tensor([[4, 5, 6], [7, 8, 9]])
print(t.size())
```

```
(2, 3)
```

```python
print(t.size(dim=1))
```

```
3
```


## Data Access

#### Tensor.data() -> memoryview

Returns the data of this tensor as a memoryview.

```python
t = Tensor([1, 2, 3, 4])
print(np.frombuffer(t.data(), dtype=np.int32))
```

```
[execution error: name 'np' is not defined]
```


#### Tensor.item() -> ConstType

Returns the value of this tensor as a standard Python number.

```python
t = Tensor(42)
print(t.item())
```

```
42
```


#### Tensor.tolist() -> Sequence[ConstType] | ConstType

Returns the value of this tensor as a nested list.
Returns single value for const tensor.

```python
t = Tensor([1, 2, 3, 4])
print(t.tolist())
```

```
[1, 2, 3, 4]
```

```python
t = Tensor(5)
print(t.tolist())
```

```
5
```


#### Tensor.numpy() -> 'numpy.ndarray'

Returns the value of this tensor as a `numpy.ndarray`.

```python
t = Tensor([1, 2, 3, 4])
print(repr(t.numpy()))
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


## tinygrad ops

#### Tensor.schedule_with_vars(*lst: 'Tensor') -> tuple[list[ExecItem], dict[str, int]]

Creates the schedule needed to realize these Tensor(s), with Variables.

NOTE: A Tensor can only be scheduled once.


#### Tensor.schedule(*lst: 'Tensor') -> list[ExecItem]

Creates the schedule needed to realize these Tensor(s).


#### Tensor.realize(*lst: 'Tensor', do_update_stats=True) -> Tensor

Triggers the computation needed to create these Tensor(s).


#### Tensor.replace(x: 'Tensor', allow_shape_mismatch=False) -> Tensor

Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.


#### Tensor.assign(x) -> Tensor


#### Tensor.detach() -> Tensor

Returns a new tensor with the same data as this tensor, but detached from the autograd graph.


#### Tensor.clone() -> Tensor

Creates a clone of this tensor allocating a separate buffer for the data.


#### Tensor.to(device: 'str | tuple[str, ...] | None') -> Tensor

Moves the tensor to the given device.


#### Tensor.to_(device: 'str | tuple[str, ...] | None') -> Tensor

Moves the tensor to the given device in place.


#### Tensor.shard(devices: 'tuple[str, ...]', axis: 'int | None' = None) -> Tensor

Shards the tensor across the given devices. Optionally specify which axis to shard on.

```python
t = Tensor.empty(2, 4)
print(t.shard((t.device, t.device), axis=1).uop)
```

```
UOp(Ops.MULTI, dtypes.float, arg=1, src=(
  UOp(Ops.SHRINK, dtypes.float, arg=None, src=(
    UOp(Ops.COPY, dtypes.float, arg=None, src=(
      UOp(Ops.RESHAPE, dtypes.float, arg=None, src=(
        UOp(Ops.BUFFER, dtypes.float, arg=8, src=(
          UOp(Ops.UNIQUE, dtypes.void, arg=606, src=()),
          UOp(Ops.DEVICE, dtypes.void, arg='CPU', src=()),)),
        UOp(Ops.VCONST, dtypes.index.vec(2), arg=(2, 4), src=()),)),
      UOp(Ops.DEVICE, dtypes.void, arg=('CPU', 'CPU'), src=()),)),
    UOp(Ops.VECTORIZE, dtypes.index.vec(2), arg=None, src=(
      UOp(Ops.CONST, dtypes.index, arg=0, src=()),
      x10:=UOp(Ops.MUL, dtypes.index, arg=None, src=(
        UOp(Ops.DEFINE_VAR, dtypes.index, arg=('_device_num', 0, 1), src=()),
        x12:=UOp(Ops.CONST, dtypes.index, arg=2, src=()),)),)),
    UOp(Ops.VECTORIZE, dtypes.index.vec(2), arg=None, src=(
      x12,
      UOp(Ops.ADD, dtypes.index, arg=None, src=(
        x10,
        x12,)),)),)),))
```


#### Tensor.shard_(devices: 'tuple[str, ...]', axis: 'int | None' = None) -> Tensor

Shards the tensor across the given devices in place.


#### Tensor.contiguous(*args, **kwargs) -> Tensor

Returns a contiguous tensor.


#### Tensor.contiguous_backward() -> Tensor

Inserts a contiguous operation in the backward pass.


## Gradient

#### Tensor.gradient(*targets: 'Tensor', gradient: 'Tensor | None' = None, materialize_grads=False) -> list[Tensor]

Computes the gradient of the targets with respect to self.

```python
x = Tensor.eye(3)
y = Tensor([[2.0,0,-2.0]])
z = y.matmul(x).sum()
dx, dy = z.gradient(x, y)

print(dx.tolist())  # dz/dx
print(dy.tolist())  # dz/dy
```

```
[[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]]
[[1.0, 1.0, 1.0]]
```


#### Tensor.backward(gradient: 'Tensor | None' = None) -> Tensor

Propagates the gradient of a tensor backwards through the computation graph.
If the 'gradient' argument is not provided, the tensor must be a scalar, and the gradient is implicitly set to 1.0.

```python
t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
t.sum().backward()
print(t.grad.numpy())
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

