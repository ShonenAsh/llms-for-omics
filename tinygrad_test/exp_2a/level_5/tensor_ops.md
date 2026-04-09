## Reduce

#### sum(axis: 'int | Sequence[int] | None' = None, keepdim=False, dtype: 'DTypeLike | None' = None) -> Tensor

Returns the sum of the elements of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the maximum is computed and whether the reduced dimensions are retained.

You can pass in `dtype` keyword argument to control the data type of the accumulation.
If not specified, the accumulation data type is chosen based on the input tensor's data type.

```python
t = Tensor.arange(6).reshape(2, 3)
print(t.numpy())
```

```
[[0 1 2]
 [3 4 5]]
```

```python
print(t.sum().numpy())
```

```
15
```

```python
print(t.sum(axis=0).numpy())
```

```
[3 5 7]
```

```python
print(t.sum(axis=1).numpy())
```

```
[ 3 12]
```


#### prod(axis: 'int | Sequence[int] | None' = None, keepdim=False, dtype: 'DTypeLike | None' = None) -> Tensor

Returns the product of the elements of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the maximum is computed and whether the reduced dimensions are retained.

You can pass in `dtype` keyword argument to control the data type of the accumulation.
If not specified, the accumulation data type is chosen based on the input tensor's data type.

```python
t = Tensor([-1, -2, -3, 1, 2, 3]).reshape(2, 3)
print(t.numpy())
```

```
[[-1 -2 -3]
 [ 1  2  3]]
```

```python
print(t.prod().numpy())
```

```
-36
```

```python
print(t.prod(axis=0).numpy())
```

```
[-1 -4 -9]
```

```python
print(t.prod(axis=1).numpy())
```

```
[-6  6]
```


#### max(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Returns the maximum value of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the maximum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```

```
[[1 0 2]
 [5 4 3]]
```

```python
print(t.max().numpy())
```

```
5
```

```python
print(t.max(axis=0).numpy())
```

```
[5 4 3]
```

```python
print(t.max(axis=1, keepdim=True).numpy())
```

```
[[2]
 [5]]
```


#### min(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Returns the minimum value of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the minimum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```

```
[[1 0 2]
 [5 4 3]]
```

```python
print(t.min().numpy())
```

```
0
```

```python
print(t.min(axis=0).numpy())
```

```
[1 0 2]
```

```python
print(t.min(axis=1, keepdim=True).numpy())
```

```
[[0]
 [3]]
```


#### any(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Tests if any element evaluates to `True` along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

```python
t = Tensor([[True, True], [True, False], [False, False]])
print(t.numpy())
```

```
[[ True  True]
 [ True False]
 [False False]]
```

```python
print(t.any().numpy())
```

```
True
```

```python
print(t.any(axis=0).numpy())
```

```
[ True  True]
```

```python
print(t.any(axis=1, keepdim=True).numpy())
```

```
[[ True]
 [ True]
 [False]]
```


#### all(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Tests if all element evaluates to `True` along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

```python
t = Tensor([[True, True], [True, False], [False, False]])
print(t.numpy())
```

```
[[ True  True]
 [ True False]
 [False False]]
```

```python
print(t.all().numpy())
```

```
False
```

```python
print(t.all(axis=0).numpy())
```

```
[False False]
```

```python
print(t.all(axis=1, keepdim=True).numpy())
```

```
[[ True]
 [False]
 [False]]
```


#### isclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> Tensor

Returns a new tensor with element-wise comparison of closeness to `other` within a tolerance.

The `rtol` and `atol` keyword arguments control the relative and absolute tolerance of the comparison.

By default, two `NaN` values are not close to each other. If `equal_nan` is `True`, two `NaN` values are considered close.

```python
print(Tensor([1e-7, 1e-8, 1e-9, float('nan')]).isclose(Tensor([0.0, 0.0, 0.0, float('nan')])).numpy())
```

```
[False  True  True False]
```

```python
print(Tensor([float('nan')]).isclose(Tensor([float('nan')]), equal_nan=True).numpy())
```

```
[ True]
```


#### allclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> bool

Check if all self and other are close. Return True or False.


#### mean(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Returns the mean value of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the mean is computed and whether the reduced dimensions are retained.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
```

```
[[2.9889 2.7339 2.7763]
 [2.3356 2.0722 2.6376]]
```

```python
print(t.mean().numpy())
```

```
2.5907671
```

```python
print(t.mean(axis=0).numpy())
```

```
[2.6623 2.4031 2.707 ]
```

```python
print(t.mean(axis=1).numpy())
```

```
[2.833  2.3485]
```


#### var(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor

Returns the variance of the tensor along the specified axis or axes.

You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
```

```
[[2.9889 2.7339 2.7763]
 [2.3356 2.0722 2.6376]]
```

```python
print(t.var().numpy())
```

```
0.10992539
```

```python
print(t.var(axis=0).numpy())
```

```
[0.2134 0.2189 0.0096]
```

```python
print(t.var(axis=1).numpy())
```

```
[0.0187 0.08  ]
```


#### var_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]

Calculates the variance and mean over the dimensions specified by dim.
Syntactic sugar around `Tensor.var` and `Tensor.mean` to match `torch.var_mean`.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
```

```
[[2.9889 2.7339 2.7763]
 [2.3356 2.0722 2.6376]]
```

```python
var, mean = t.var_mean()
print(var.numpy(), mean.numpy())
```

```
0.10992539 2.5907671
```


#### std(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor

Returns the standard deviation of the tensor along the specified axis or axes.

You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
which the standard deviation is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
```

```
[[2.9889 2.7339 2.7763]
 [2.3356 2.0722 2.6376]]
```

```python
print(t.std().numpy())
```

```
0.33154997
```

```python
print(t.std(axis=0).numpy())
```

```
[0.462  0.4679 0.0981]
```

```python
print(t.std(axis=1).numpy())
```

```
[0.1367 0.2829]
```


#### std_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]

Calculates the standard deviation and mean over the dimensions specified by dim.
Syntactic sugar around `Tensor.std` and `Tensor.mean` to match `torch.std_mean`.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
```

```
[[2.9889 2.7339 2.7763]
 [2.3356 2.0722 2.6376]]
```

```python
std, mean = t.std_mean()
print(std.numpy(), mean.numpy())
```

```
0.33154997 2.5907671
```


#### softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor

Applies the softmax function to the tensor along the specified axis.

Rescales the elements of the tensor such that they lie in the range [0, 1] and sum to 1.

You can pass in the `axis` keyword argument to control the axis along which the softmax is computed.

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 3)
print(t.numpy())
```

```
[[ 0.9779  0.4678  0.5526]
 [-0.3288 -0.8555  0.2753]]
```

```python
print(t.softmax().numpy())
```

```
[[0.4436 0.2664 0.29  ]
 [0.2924 0.1727 0.5349]]
```

```python
print(t.softmax(axis=0).numpy())
```

```
[[0.787  0.7897 0.5689]
 [0.213  0.2103 0.4311]]
```


#### log_softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor

Applies the log-softmax function to the tensor along the specified axis.

The log-softmax function is a numerically stable alternative to the softmax function in log space.

You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 3)
print(t.numpy())
```

```
[[ 0.9779  0.4678  0.5526]
 [-0.3288 -0.8555  0.2753]]
```

```python
print(t.log_softmax().numpy())
```

```
[[-0.8127 -1.3228 -1.238 ]
 [-1.2297 -1.7564 -0.6256]]
```

```python
print(t.log_softmax(axis=0).numpy())
```

```
[[-0.2396 -0.2361 -0.564 ]
 [-1.5463 -1.5594 -0.8414]]
```


#### logsumexp(axis=None, keepdim=False) -> Tensor

Computes the log-sum-exp of the tensor along the specified axis or axes.

The log-sum-exp function is a numerically stable way to compute the logarithm of the sum of exponentials.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the log-sum-exp is computed and whether the reduced dimensions are retained.

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 3)
print(t.numpy())
```

```
[[ 0.9779  0.4678  0.5526]
 [-0.3288 -0.8555  0.2753]]
```

```python
print(t.logsumexp().numpy())
```

```
2.1347282
```

```python
print(t.logsumexp(axis=0).numpy())
```

```
[1.2174 0.7039 1.1167]
```

```python
print(t.logsumexp(axis=1).numpy())
```

```
[1.7906 0.9009]
```


#### logcumsumexp(axis=0) -> Tensor

Computes the log-cumsum-exp of the tensor along the specified axis or axes.

The log-cumsum-exp function is a numerically stable way to compute the logarithm of the cumulative sum of exponentials.

You can pass in the `axis` keyword argument to control the axis along which
the log-cumsum-exp is computed.

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 3)
print(t.numpy())
```

```
[[ 0.9779  0.4678  0.5526]
 [-0.3288 -0.8555  0.2753]]
```

```python
print(t.logcumsumexp().numpy())
```

```
[[0.9779 0.4678 0.5526]
 [1.2174 0.7039 1.1167]]
```

```python
print(t.logcumsumexp(axis=0).numpy())
```

```
[[0.9779 0.4678 0.5526]
 [1.2174 0.7039 1.1167]]
```

```python
print(t.logcumsumexp(axis=1).numpy())
```

```
[[ 0.9779  1.4481  1.7906]
 [-0.3288  0.1353  0.9009]]
```


#### argmax(axis=None, keepdim=False) -> Tensor

Returns the indices of the maximum value of the tensor along the specified axis.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the maximum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```

```
[[1 0 2]
 [5 4 3]]
```

```python
print(t.argmax().numpy()) # Returns the index of the maximum value in the flattened tensor.
```

```
3
```

```python
print(t.argmax(axis=0).numpy()) # Returns the indices of the maximum values along axis 0.
```

```
[1 1 1]
```

```python
print(t.argmax(axis=1).numpy()) # Returns the indices of the maximum values along axis 1.
```

```
[2 0]
```


#### argmin(axis=None, keepdim=False) -> Tensor

Returns the indices of the minimum value of the tensor along the specified axis.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the minimum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```

```
[[1 0 2]
 [5 4 3]]
```

```python
print(t.argmin().numpy()) # Returns the index of the minimum value in the flattened tensor.
```

```
1
```

```python
print(t.argmin(axis=0).numpy()) # Returns the indices of the minimum values along axis 0.
```

```
[0 0 0]
```

```python
print(t.argmin(axis=1).numpy()) # Returns the indices of the minimum values along axis 1.
```

```
[1 2]
```


## Processing

#### avg_pool2d(kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, ceil_mode=False, count_include_pad=True) -> Tensor

Applies average pooling over a tensor.

This function supports three different types of `padding`

1. `int` (single value):
  Applies the same padding value uniformly to all spatial dimensions.

2. `tuple[int, ...]` (length = number of spatial dimensions):
  Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
  Specifies explicit padding for each side of each spatial dimension in the form
  `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

When `ceil_mode` is set to `True`, output shape will be determined using ceil division.
When `count_include_pad` is set to `False`, zero padding will not be included in the averaging calculation.

NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

```python
t = Tensor.arange(25).reshape(1, 1, 5, 5)
print(t.avg_pool2d().numpy())
```

```
[[[[ 3.  5.]
   [13. 15.]]]]
```

```python
print(t.avg_pool2d(ceil_mode=True).numpy())
```

```
[[[[ 3.   5.   6.5]
   [13.  15.  16.5]
   [20.5 22.5 24. ]]]]
```

```python
print(t.avg_pool2d(padding=1).numpy())
```

```
[[[[ 0.    0.75  1.75]
   [ 3.75  9.   11.  ]
   [ 8.75 19.   21.  ]]]]
```

```python
print(t.avg_pool2d(padding=1, count_include_pad=False).numpy())
```

```
[[[[ 0.   1.5  3.5]
   [ 7.5  9.  11. ]
   [17.5 19.  21. ]]]]
```


#### max_pool2d(kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, ceil_mode=False, return_indices=False) -> Tensor | tuple[Tensor, Tensor]

Applies max pooling over a tensor.

This function supports three different types of `padding`

1. `int` (single value):
  Applies the same padding value uniformly to all spatial dimensions.

2. `tuple[int, ...]` (length = number of spatial dimensions):
  Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
  Specifies explicit padding for each side of each spatial dimension in the form
  `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

When `ceil_mode` is set to `True`, output shape will be determined using ceil division.
When `return_indices` is set to `True`, the argmax will be returned along with the max values.

NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

```python
t = Tensor.arange(25).reshape(1, 1, 5, 5)
print(t.max_pool2d().numpy())
```

```
[[[[ 6  8]
   [16 18]]]]
```

```python
print(t.max_pool2d(ceil_mode=True).numpy())
```

```
[[[[ 6  8  9]
   [16 18 19]
   [21 23 24]]]]
```

```python
print(t.max_pool2d(padding=1).numpy())
```

```
[[[[ 0  2  4]
   [10 12 14]
   [20 22 24]]]]
```


#### max_unpool2d(indices: 'Tensor', kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, output_size=None)

Performs a partial inverse of `max_pool2d` using the indices from the argmax.

When `output_size` is provided, the output shape disambiguates to the provided shape.

NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

```python
t = Tensor.arange(1, 17).reshape(1, 1, 4, 4)
print(t.numpy())
```

```
[[[[ 1  2  3  4]
   [ 5  6  7  8]
   [ 9 10 11 12]
   [13 14 15 16]]]]
```

```python
output, indices = Tensor.max_pool2d(t, return_indices=True)
print(output.numpy())
print(indices.numpy())
```

```
[[[[ 6  8]
   [14 16]]]]
[[[[ 5  7]
   [13 15]]]]
```

```python
print(Tensor.max_unpool2d(output, indices).numpy())
```

```
[[[[ 0  0  0  0]
   [ 0  6  0  8]
   [ 0  0  0  0]
   [ 0 14  0 16]]]]
```


#### conv2d(weight: 'Tensor', bias: 'Tensor | None' = None, groups=1, stride=1, dilation=1, padding: 'int | tuple[int, ...]' = 0, dtype: 'DTypeLike | None' = None) -> Tensor

Applies a convolution over a tensor with a given `weight` and optional `bias`.

This function supports three different types of `padding`

1. `int` (single value):
  Applies the same padding value uniformly to all spatial dimensions.

2. `tuple[int, ...]` (length = number of spatial dimensions):
  Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
  Specifies explicit padding for each side of each spatial dimension in the form
  `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

```python
t = Tensor.arange(9).reshape(1, 1, 3, 3)
w = Tensor.ones(1, 1, 2, 2)
print(t.conv2d(w).numpy())
```

```
[[[[ 8. 12.]
   [20. 24.]]]]
```


#### conv_transpose2d(weight: 'Tensor', bias: 'Tensor | None' = None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor

Applies a transposed convolution over a tensor with a given `weight` and optional `bias`.

This function supports three different types of `padding`

1. `int` (single value):
  Applies the same padding value uniformly to all spatial dimensions.

2. `tuple[int, ...]` (length = number of spatial dimensions):
  Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.

3. `tuple[int, ...]` (length = 2 * number of spatial dimensions):
  Specifies explicit padding for each side of each spatial dimension in the form
  `(padding_left, padding_right, padding_top, padding_bottom, ...)`.

NOTE: unlike PyTorch, this implementation is not limited to only 2d transposed convolutions and instead works for any number of dimensions.

See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

```python
t = Tensor.arange(9).reshape(1, 1, 3, 3)
w = Tensor.ones(1, 1, 2, 2)
print(t.conv_transpose2d(w).numpy())
```

```
[[[[ 0.  1.  3.  2.]
   [ 3.  8. 12.  7.]
   [ 9. 20. 24. 13.]
   [ 6. 13. 15.  8.]]]]
```


#### dot(w: 'Tensor', dtype: 'DTypeLike | None' = None) -> Tensor

Performs dot product between two tensors.
If `w` is 1-D, it's a sum product over the last axis of `self` and `w`.
If `w` is N-D with N>=2, it's a sum product over the last axis of `self` and the second-to-last axis of `w`.

You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

```python
a = Tensor([1, 2, 3])
b = Tensor([1, 1, 0])
print(a.dot(b).numpy())
```

```
3
```

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])
print(a.dot(b).numpy())
```

```
[[19 22]
 [43 50]]
```


#### matmul(x: 'Tensor', reverse=False, dtype: 'DTypeLike | None' = None) -> Tensor

Performs matrix multiplication between two tensors.

You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])
print(a.matmul(b).numpy())
```

```
[[19 22]
 [43 50]]
```


#### einsum(formula: 'str', *operands: 'Tensor | Sequence[Tensor]', dtype: 'DTypeLike | None' = None) -> Tensor

Sums the product of the elements of the input tensors according to a formula based on the Einstein summation convention.

See: https://pytorch.org/docs/stable/generated/torch.einsum.html

```python
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])
print(Tensor.einsum("ij,ij->", x, y).numpy())
```

```
70
```


#### cumsum(axis: 'int' = 0) -> Tensor

Computes the cumulative sum of the tensor along the specified `axis`.

```python
t = Tensor.ones(2, 3)
print(t.numpy())
```

```
[[1. 1. 1.]
 [1. 1. 1.]]
```

```python
print(t.cumsum(1).numpy())
```

```
[[1. 2. 3.]
 [1. 2. 3.]]
```


#### cumprod(axis: 'int') -> Tensor

Computes the cumulative product of the elements of the tensor along the specified `axis`.

```python
t = Tensor.arange(1, 7).reshape(2, 3)
print(t.numpy())
```

```
[[1 2 3]
 [4 5 6]]
```

```python
print(t.cumprod(axis=0).numpy())
```

```
[[ 1  2  3]
 [ 4 10 18]]
```


#### cummax(axis: 'int' = 0) -> Tensor

Computes the cumulative max of the tensor along the specified `axis`.

```python
t = Tensor([0, 1, -1, 2, -2, 3, -3])
print(t.numpy())
```

```
[ 0  1 -1  2 -2  3 -3]
```

```python
print(t.cummax(0).numpy())
```

```
[0 1 1 2 2 3 3]
```


#### cummin *(resolution failed)*

#### triu(diagonal: 'int' = 0) -> Tensor

Returns the upper triangular part of the tensor, the other elements are set to 0.

The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

```python
t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(t.numpy())
```

```
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```

```python
print(t.triu(diagonal=0).numpy())
```

```
[[ 1  2  3  4]
 [ 0  6  7  8]
 [ 0  0 11 12]]
```

```python
print(t.triu(diagonal=1).numpy())
```

```
[[ 0  2  3  4]
 [ 0  0  7  8]
 [ 0  0  0 12]]
```

```python
print(t.triu(diagonal=-1).numpy())
```

```
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 0 10 11 12]]
```


#### tril(diagonal: 'int' = 0) -> Tensor

Returns the lower triangular part of the tensor, the other elements are set to 0.

The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

```python
t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(t.numpy())
```

```
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```

```python
print(t.tril(diagonal=0).numpy())
```

```
[[ 1  0  0  0]
 [ 5  6  0  0]
 [ 9 10 11  0]]
```

```python
print(t.tril(diagonal=1).numpy())
```

```
[[ 1  2  0  0]
 [ 5  6  7  0]
 [ 9 10 11 12]]
```

```python
print(t.tril(diagonal=-1).numpy())
```

```
[[ 0  0  0  0]
 [ 5  0  0  0]
 [ 9 10  0  0]]
```


#### interpolate(size: 'tuple[int, ...]', mode: 'str' = 'linear', align_corners: 'bool' = False) -> Tensor

Downsamples or Upsamples to the input `size`, accepts 0 to N batch dimensions.

The interpolation algorithm is selected with `mode` which currently only supports `linear`, `nearest` and `nearest-exact`.
To run `bilinear` or `trilinear`, pass in a 2D or 3D size.

```python
t = Tensor([[1, 2, 3, 4], [21, 22, 23, 24], [41, 42, 43, 44]])
print(t.numpy())
```

```
[[ 1  2  3  4]
 [21 22 23 24]
 [41 42 43 44]]
```

```python
print(t.interpolate(size=(2,3), mode="linear").numpy())
```

```
[[ 6  7  8]
 [36 37 38]]
```


#### scatter(dim: 'int', index: 'Tensor', src: 'Tensor | ConstType', reduce: "Literal['multiply', 'add'] | None" = None) -> Tensor

Scatters `src` values along an axis specified by `dim`.
Apply `add` or `multiply` reduction operation with `reduce`.

NOTE: To use the `reduce` argument with a Tensor `src`, see `Tensor.scatter_reduce`.

```python
src = Tensor.arange(1, 11).reshape(2, 5)
print(src.numpy())
```

```
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
```

```python
index = Tensor([[0, 1, 2, 0]])
print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(0, index, src).numpy())
```

```
[[1 0 0 4 0]
 [0 2 0 0 0]
 [0 0 3 0 0]]
```

```python
index = Tensor([[0, 1, 2], [0, 1, 4]])
print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(1, index, src).numpy())
```

```
[[1 2 3 0 0]
 [6 7 0 0 8]
 [0 0 0 0 0]]
```

```python
print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='multiply').numpy())
```

```
[[2.   2.   2.46 2.  ]
 [2.   2.   2.   2.46]]
```

```python
print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='add').numpy())
```

```
[[2.   2.   3.23 2.  ]
 [2.   2.   2.   3.23]]
```


#### scatter_reduce(dim: 'int', index: 'Tensor', src: 'Tensor', reduce: "Literal['sum', 'prod', 'mean', 'amax', 'amin']", include_self: 'bool' = True) -> Tensor

Scatters `src` values along an axis specified by `dim`.
Apply `"sum"`, `"prod"`, `"mean"`, `"amax"`, or `"amin"` reduction operations with `reduce`.

Set `include_self=False` to exclude values in the `self` Tensor from the reduction.

```python
src = Tensor.arange(1, 11).cast(dtypes.float).reshape(2, 5)
print(src.numpy())
index = Tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
print(index.numpy())
```

```
[[ 1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10.]]
[[0 0 0 0 0]
 [0 0 0 0 0]]
```

```python
print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='sum').numpy())
```

```
[[ 8. 10. 12. 14. 16.]]
```

```python
print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='prod').numpy())
```

```
[[ 6. 14. 24. 36. 50.]]
```

```python
print(Tensor.ones(1, 5, dtype=src.dtype).scatter_reduce(0, index, src, reduce='mean', include_self=False).numpy())
```

```
[[3.5 4.5 5.5 6.5 7.5]]
```

```python
print(Tensor([[-10, 20, 0, 5, 10]], dtype=src.dtype).scatter_reduce(0, index, src, reduce='amax').numpy())
```

```
[[ 6. 20.  8.  9. 10.]]
```

```python
print(Tensor([[-10, 20, 0, 5, 10]], dtype=src.dtype).scatter_reduce(0, index, src, reduce='amin').numpy())
```

```
[[-10.   2.   0.   4.   5.]]
```


#### masked_select(mask)

Selects elements from `self` based on the boolean `mask`.

```python
t = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
mask = Tensor([[True, False, True], [False, True, False], [False, False, True]])
print(t.numpy())
print(mask.numpy())
```

```
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[ True False  True]
 [False  True False]
 [False False  True]]
```

```python
print(t.masked_select(mask).numpy())
```

```
[0 2 4 8]
```


#### masked_fill(mask: 'Tensor', value: 'Tensor | ConstType') -> Tensor

Replaces `self` with `value` wherever the elements of `mask` are True.

```python
t = Tensor([1, 2, 3, 4, 5])
mask = Tensor([True, False, True, False, False])
print(t.masked_fill(mask, -12).numpy())
```

```
[-12   2 -12   4   5]
```

```python
t = Tensor([1, 2, 3, 4, 5])
mask = Tensor([True, False, True, False, False])
value = Tensor([-1, -2, -3, -4, -5])
print(t.masked_fill(mask, value).numpy())
```

```
[-1  2 -3  4  5]
```


#### nonzero *(resolution failed)*

#### sort(dim: 'int' = -1, descending: 'bool' = False) -> tuple[Tensor, Tensor]

Performs a bitonic sort on the tensor along the specified dimension.

Order of indices for equivalent elements is always preserved.

See: https://en.wikipedia.org/wiki/Bitonic_sorter

```python
t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
print(t.numpy())
```

```
[[0.1 0.5 1.2 3.4 2.1]
 [2.2 1.9 0.3 4.5 0.8]]
```

```python
sorted_values, indices = t.sort(dim=1, descending=True)
print(sorted_values.numpy())
print(indices.numpy())
```

```
[[3.4 2.1 1.2 0.5 0.1]
 [4.5 2.2 1.9 0.8 0.3]]
[[3 4 2 1 0]
 [3 0 1 4 2]]
```


#### argsort(dim: 'int' = -1, descending: 'bool' = False) -> Tensor

Returns the indices that sort input tensor along given `dimension` in given `descending` order by value.

```python
t = Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
print(t.argsort().numpy())
```

```
[[3 0 1 2]
 [0 3 2 1]]
```


#### topk(k: 'int', dim: 'int' = -1, largest: 'bool' = True, sorted_: 'bool' = True) -> tuple[Tensor, Tensor]

Computes the top-k elements of the tensor along the specified `dim`.

Order of indices for equivalent elements is always preserved.

```python
t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
print(t.numpy())
```

```
[[0.1 0.5 1.2 3.4 2.1]
 [2.2 1.9 0.3 4.5 0.8]]
```

```python
topk_values, topk_indices = t.topk(2, dim=1)
print(topk_values.numpy())
print(topk_indices.numpy())
```

```
[[3.4 2.1]
 [4.5 2.2]]
[[3 4]
 [3 0]]
```


#### multinomial(num_samples: 'int' = 1, replacement: 'bool' = False) -> Tensor

Returns a tensor with `num_samples` indices sampled from a multinomial distribution weighted by `self`.

NOTE: `replacement=False` for `num_samples > 1` is not supported yet.

```python
Tensor.manual_seed(42)
t = Tensor([1, 2, 3, 4])
print(t.multinomial(20, replacement=True).numpy())
```

```
[2 1 3 2 3 1 2 2 3 3 3 3 3 3 2 3 2 3 3 3]
```


## Neural Network (functional)

#### linear(weight: 'Tensor', bias: 'Tensor | None' = None, dtype: 'DTypeLike | None' = None) -> Tensor

Applies a linear transformation to `self` using `weight` and `bias`.

See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

```python
t = Tensor([[1, 2], [3, 4]])
weight = Tensor([[1, 2], [3, 4]])
bias = Tensor([1, 2])
print(t.linear(weight, bias).numpy())
```

```
[[ 8 12]
 [16 24]]
```


#### sequential(ll: 'list[Callable[[Tensor], Tensor]]') -> Tensor

Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

```python
t = Tensor([1, 2, 3])
print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
```

```
[3 5 7]
```


#### layernorm(axis: 'int | tuple[int, ...]' = -1, eps: 'float' = 1e-05) -> Tensor

Applies Layer Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1607.06450v1

```python
t = Tensor.randn(8, 10, 16) * 2 + 8
print(t.mean().item(), t.std().item())
```

```
7.9793524742126465 2.074720621109009
```

```python
t = t.layernorm()
print(t.mean().item(), t.std().item())
```

```
7.269673196752535e-10 1.0003894567489624
```


#### batchnorm(weight: 'Tensor | None', bias: 'Tensor | None', mean: 'Tensor', invstd: 'Tensor', axis: 'int | tuple[int, ...]' = 1) -> Tensor

Applies Batch Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1502.03167

```python
t = Tensor.randn(8, 4, 16, 16) * 2 + 8
print(t.mean().item(), t.std().item())
```

```
8.019729614257812 1.9927232265472412
```

```python
t = t.batchnorm(None, None, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
print(t.mean().item(), t.std().item())
```

```
6.119149134065083e-07 0.9998146891593933
```


#### dropout(p=0.5) -> Tensor

Applies dropout to `self`.

NOTE: dropout is only applied when `Tensor.training` is `True`.

- Paper: https://jmlr.org/papers/v15/srivastava14a.html

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 2)
with Tensor.train():
  print(t.dropout().numpy())
```

```
[[-1.0287  2.17  ]
 [ 1.8178  0.    ]]
```


#### one_hot(num_classes: 'int' = -1) -> Tensor

Converts `self` to a one-hot tensor.

`num_classes` defaults to -1, which means num_classes will be inferred as max(self) + 1.

```python
t = Tensor([0, 1, 3, 3, 4])
print(t.one_hot(5).numpy())
```

```
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 0 1 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
```


#### scaled_dot_product_attention(key: 'Tensor', value: 'Tensor', attn_mask: 'Tensor | None' = None, dropout_p: 'float' = 0.0, is_causal: 'bool' = False, enable_gqa: 'bool' = False) -> Tensor

Computes scaled dot-product attention.
`self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

- Paper: https://arxiv.org/abs/1706.03762v7

```python
q = Tensor.randn(2, 4, 8)
k = Tensor.randn(2, 4, 8)
v = Tensor.randn(2, 4, 8)
print(q.scaled_dot_product_attention(k, v).numpy())
```

```
[[[ 0.6408  0.3264  0.7317 -1.0943  0.5778 -0.0534 -0.0104 -0.0488]
  [ 0.1243 -0.8259  1.6481 -0.8035 -0.3961  0.4269  0.1232  1.6462]
  [ 0.9535  0.1068  0.8545 -0.5395  0.4692 -0.0548 -0.2274  0.6152]
  [ 0.8891 -0.0411  0.7818 -0.3322  0.3931 -0.0202 -0.1101  0.8129]]

 [[-0.4273 -0.6085 -0.0465  0.5246  0.3641 -0.0381 -0.0106  0.8349]
  [ 0.6321  0.3654  0.4137 -0.2327  0.2558  0.1418 -1.27   -0.802 ]
  [ 0.1794  0.4616  0.1847 -0.1988  0.2123  0.1837 -0.9583 -0.5364]
  [ 0.4408  0.6125  0.0811 -0.3886  0.3602  0.4987 -1.4414 -0.9565]]]
```


#### binary_crossentropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean') -> Tensor

Computes the binary cross-entropy loss between `self` and `Y`.

See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

```python
t = Tensor([0.1, 0.9, 0.2])
Y = Tensor([0, 1, 0])
print(t.binary_crossentropy(Y).item())
```

```
0.14462155103683472
```


#### binary_crossentropy_logits(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', pos_weight: 'Tensor | None' = None) -> Tensor

Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.

See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

```python
t = Tensor([-1, 2, -3])
Y = Tensor([0, 1, 0])
print(t.binary_crossentropy_logits(Y).item())
```

```
0.16292566061019897
```


#### sparse_categorical_crossentropy(Y: 'Tensor', ignore_index: 'int' = -1, label_smoothing=0.0, reduction: 'ReductionStr' = 'mean') -> Tensor

Computes the sparse categorical cross-entropy loss between `self` and `Y`.

NOTE: `self` is logits and `Y` is the target labels.
NOTE: unlike PyTorch, this function expects the class axis to be -1

See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

```python
t = Tensor([[-1, 2, -3], [1, -2, 3]])
Y = Tensor([1, 2])
print(t.sparse_categorical_crossentropy(Y).item())
```

```
0.09391524642705917
```


#### cross_entropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', label_smoothing: 'float' = 0.0) -> Tensor

Computes the cross entropy loss between input logits and target.

NOTE: `self` are logits and `Y` are the target labels or class probabilities.

See: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

```python
t = Tensor([[-1, 2, -3], [1, -2, 3]])
Y = Tensor([1, 2])
print(t.cross_entropy(Y).item())
```

```
0.09391524642705917
```

```python
t = Tensor([[-1, 2, -3], [1, -2, 3]])
Y = Tensor([1, 2])
print(t.cross_entropy(Y, reduction='none').numpy())
```

```
[0.055  0.1328]
```


#### nll_loss(Y: 'Tensor', weight: 'Tensor | None' = None, ignore_index: 'int | None' = None, reduction: 'ReductionStr' = 'mean') -> Tensor

Computes the negative log likelihood loss between log-probabilities and target labels.

NOTE: `self` is log-probabilities and `Y` is the Y labels or class probabilities.

See: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html

```python
t = Tensor([[-1, 2, -3], [1, -2, 3]])
Y = Tensor([1, 2])
print(t.log_softmax().nll_loss(Y).item())
```

```
0.09391524642705917
```

```python
t = Tensor([[-1, 2, -3], [1, -2, 3]])
Y = Tensor([1, 2])
print(t.log_softmax().nll_loss(Y, reduction='none').numpy())
```

```
[0.055  0.1328]
```


## Linear Algebra

#### qr() -> tuple[Tensor, Tensor]


#### svd(full_matrices=True) -> tuple[Tensor, Tensor, Tensor]

