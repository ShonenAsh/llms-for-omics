## Creation (basic)

#### empty(*shape, device: 'str | tuple[str, ...] | None' = None, dtype: 'DTypeLike | None' = None, **kwargs) -> Tensor

Creates an empty tensor with the given shape.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.empty(2, 3)
print(t.shape)
```


#### zeros(*shape, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with zeros.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
print(Tensor.zeros(2, 3).numpy())
```


#### ones(*shape, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with ones.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
print(Tensor.ones(2, 3).numpy())
```


#### full(shape: 'tuple[sint, ...]', fill_value: 'ConstType', **kwargs) -> Tensor

Creates a tensor with the given shape, filled with the given value.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
print(Tensor.full((2, 3), 42).numpy())
```


#### arange(start, stop=None, step=1, **kwargs) -> Tensor

Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.

If `stop` is not specified, values are generated from `[0, start)` with the given `step`.

If `stop` is specified, values are generated from `[start, stop)` with the given `step`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
print(Tensor.arange(5).numpy())
```


#### linspace(start: 'int | float', stop: 'int | float', steps: 'int', **kwargs) -> Tensor

Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
print(Tensor.linspace(0, 10, 5).numpy())
```


#### eye(n: 'int', m: 'int | None' = None, dtype=None, device=None, requires_grad: 'bool | None' = None) -> Tensor

Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
print(Tensor.eye(3).numpy())
```


#### full_like(fill_value: 'ConstType', **kwargs) -> Tensor

Creates a tensor with the same shape as `self`, filled with the given value.
If `dtype` is not specified, the dtype of `self` is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.ones(2, 3)
print(Tensor.full_like(t, 42).numpy())
```


#### zeros_like(**kwargs) -> Tensor

Creates a tensor with the same shape as `self`, filled with zeros.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.ones(2, 3)
print(Tensor.zeros_like(t).numpy())
```


#### ones_like(**kwargs) -> Tensor

Creates a tensor with the same shape as `self`, filled with ones.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.zeros(2, 3)
print(Tensor.ones_like(t).numpy())
```


## Creation (external)

#### from_blob(ptr: 'int', shape: 'tuple[int, ...]', **kwargs) -> Tensor

Exposes the pointer as a Tensor without taking ownership of the original data.
The pointer must remain valid for the entire lifetime of the created Tensor.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.


#### from_url(url: 'str', gunzip: 'bool' = False, **kwargs) -> Tensor

Creates a Tensor from a URL.

This is the preferred way to access Internet resources.
It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
This also will soon become lazy (when possible) and not print progress without DEBUG.

The `gunzip` flag will gzip extract the resource and return an extracted Tensor.


## Creation (random)

#### manual_seed(seed=0) -> None

Sets the seed for random operations.

```python
Tensor.manual_seed(42)
print(Tensor.rand(5).numpy())
print(Tensor.rand(5).numpy())
```


#### rand(*shape, device: 'str | None' = None, dtype: 'DTypeLike | None' = None, contiguous: 'bool' = True, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
t = Tensor.rand(2, 3)
print(t.numpy())
```


#### rand_like(**kwargs) -> Tensor

Creates a tensor with the same shape and sharding as `self`, filled with random values from a uniform distribution over the interval `[0, 1)`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.ones(2, 3)
print(Tensor.rand_like(t).numpy())
```


#### randn(*shape, dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
If `dtype` is not specified, the default type is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.randn(2, 3).numpy())
```


#### randn_like(dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor

Creates a tensor with the same shape and sharding as `self`, filled with random values from a normal distribution with mean 0 and variance 1.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
t = Tensor.ones(2, 3)
print(Tensor.randn_like(t).numpy())
```


#### randint(*shape, low=0, high=10, dtype=dtypes.int, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
If `dtype` is not specified, the default type is used.

You can pass in the `device` keyword argument to control device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.randint(2, 3, low=5, high=10).numpy())
```


#### randperm(n: 'int', device=None, dtype=dtypes.int, **kwargs) -> Tensor

Returns a tensor with a random permutation of integers from `0` to `n-1`.

```python
Tensor.manual_seed(42)
print(Tensor.randperm(6).numpy())
```


#### normal(*shape, mean=0.0, std=1.0, requires_grad: 'bool | None' = None, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.normal(2, 3, mean=10, std=2).numpy())
```


#### uniform(*shape, low=0.0, high=1.0, dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.uniform(2, 3, low=2, high=10).numpy())
```


#### scaled_uniform(*shape, **kwargs) -> Tensor

Creates a tensor with the given shape, filled with random values from a uniform distribution
over the interval `[-prod(shape)**-0.5, prod(shape)**-0.5)`.

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.scaled_uniform(2, 3).numpy())
```


#### glorot_uniform(*shape, **kwargs) -> Tensor

<https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.glorot_uniform(2, 3).numpy())
```


#### kaiming_uniform(*shape, a: 'float' = 0.01, **kwargs) -> Tensor

<https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.kaiming_uniform(2, 3).numpy())
```


#### kaiming_normal(*shape, a: 'float' = 0.01, **kwargs) -> Tensor

<https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
Additionally, all other keyword arguments are passed to the constructor of the tensor.

```python
Tensor.manual_seed(42)
print(Tensor.kaiming_normal(2, 3).numpy())
```

