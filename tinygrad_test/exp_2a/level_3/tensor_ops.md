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


#### max(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Returns the maximum value of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the maximum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```


#### min(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Returns the minimum value of the tensor along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the minimum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```


#### any(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Tests if any element evaluates to `True` along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

```python
t = Tensor([[True, True], [True, False], [False, False]])
print(t.numpy())
```


#### all(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor

Tests if all element evaluates to `True` along the specified axis or axes.

You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

```python
t = Tensor([[True, True], [True, False], [False, False]])
print(t.numpy())
```


#### isclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> Tensor

Returns a new tensor with element-wise comparison of closeness to `other` within a tolerance.

The `rtol` and `atol` keyword arguments control the relative and absolute tolerance of the comparison.

By default, two `NaN` values are not close to each other. If `equal_nan` is `True`, two `NaN` values are considered close.

```python
print(Tensor([1e-7, 1e-8, 1e-9, float('nan')]).isclose(Tensor([0.0, 0.0, 0.0, float('nan')])).numpy())
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


#### var(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor

Returns the variance of the tensor along the specified axis or axes.

You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
```


#### var_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]

Calculates the variance and mean over the dimensions specified by dim.
Syntactic sugar around `Tensor.var` and `Tensor.mean` to match `torch.var_mean`.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
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


#### std_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]

Calculates the standard deviation and mean over the dimensions specified by dim.
Syntactic sugar around `Tensor.std` and `Tensor.mean` to match `torch.std_mean`.

```python
Tensor.manual_seed(42)
t = Tensor.normal(2, 3, mean=2.5, std=0.5)
print(t.numpy())
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


#### log_softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor

Applies the log-softmax function to the tensor along the specified axis.

The log-softmax function is a numerically stable alternative to the softmax function in log space.

You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

```python
Tensor.manual_seed(42)
t = Tensor.randn(2, 3)
print(t.numpy())
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


#### argmax(axis=None, keepdim=False) -> Tensor

Returns the indices of the maximum value of the tensor along the specified axis.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the maximum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
```


#### argmin(axis=None, keepdim=False) -> Tensor

Returns the indices of the minimum value of the tensor along the specified axis.

You can pass in `axis` and `keepdim` keyword arguments to control the axis along
which the minimum is computed and whether the reduced dimensions are retained.

```python
t = Tensor([[1, 0, 2], [5, 4, 3]])
print(t.numpy())
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


#### max_unpool2d(indices: 'Tensor', kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, output_size=None)

Performs a partial inverse of `max_pool2d` using the indices from the argmax.

When `output_size` is provided, the output shape disambiguates to the provided shape.

NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

```python
t = Tensor.arange(1, 17).reshape(1, 1, 4, 4)
print(t.numpy())
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


#### matmul(x: 'Tensor', reverse=False, dtype: 'DTypeLike | None' = None) -> Tensor

Performs matrix multiplication between two tensors.

You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
You can pass in the optional `dtype` keyword argument to control the data type of the accumulation.

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])
print(a.matmul(b).numpy())
```


#### einsum(formula: 'str', *operands: 'Tensor | Sequence[Tensor]', dtype: 'DTypeLike | None' = None) -> Tensor

Sums the product of the elements of the input tensors according to a formula based on the Einstein summation convention.

See: https://pytorch.org/docs/stable/generated/torch.einsum.html

```python
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])
print(Tensor.einsum("ij,ij->", x, y).numpy())
```


#### cumsum(axis: 'int' = 0) -> Tensor

Computes the cumulative sum of the tensor along the specified `axis`.

```python
t = Tensor.ones(2, 3)
print(t.numpy())
```


#### cumprod(axis: 'int') -> Tensor

Computes the cumulative product of the elements of the tensor along the specified `axis`.

```python
t = Tensor.arange(1, 7).reshape(2, 3)
print(t.numpy())
```


#### cummax(axis: 'int' = 0) -> Tensor

Computes the cumulative max of the tensor along the specified `axis`.

```python
t = Tensor([0, 1, -1, 2, -2, 3, -3])
print(t.numpy())
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


#### tril(diagonal: 'int' = 0) -> Tensor

Returns the lower triangular part of the tensor, the other elements are set to 0.

The argument `diagonal` determines which diagonal is on the boundary. `diagonal = 0` means the main diagonal.
Positive `diagonal` means above the main diagonal, and negative `diagonal` means below the main diagonal.

```python
t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(t.numpy())
```


#### interpolate(size: 'tuple[int, ...]', mode: 'str' = 'linear', align_corners: 'bool' = False) -> Tensor

Downsamples or Upsamples to the input `size`, accepts 0 to N batch dimensions.

The interpolation algorithm is selected with `mode` which currently only supports `linear`, `nearest` and `nearest-exact`.
To run `bilinear` or `trilinear`, pass in a 2D or 3D size.

```python
t = Tensor([[1, 2, 3, 4], [21, 22, 23, 24], [41, 42, 43, 44]])
print(t.numpy())
```


#### scatter(dim: 'int', index: 'Tensor', src: 'Tensor | ConstType', reduce: "Literal['multiply', 'add'] | None" = None) -> Tensor

Scatters `src` values along an axis specified by `dim`.
Apply `add` or `multiply` reduction operation with `reduce`.

NOTE: To use the `reduce` argument with a Tensor `src`, see `Tensor.scatter_reduce`.

```python
src = Tensor.arange(1, 11).reshape(2, 5)
print(src.numpy())
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


#### masked_select(mask)

Selects elements from `self` based on the boolean `mask`.

```python
t = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
mask = Tensor([[True, False, True], [False, True, False], [False, False, True]])
print(t.numpy())
print(mask.numpy())
```


#### masked_fill(mask: 'Tensor', value: 'Tensor | ConstType') -> Tensor

Replaces `self` with `value` wherever the elements of `mask` are True.

```python
t = Tensor([1, 2, 3, 4, 5])
mask = Tensor([True, False, True, False, False])
print(t.masked_fill(mask, -12).numpy())
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


#### argsort(dim: 'int' = -1, descending: 'bool' = False) -> Tensor

Returns the indices that sort input tensor along given `dimension` in given `descending` order by value.

```python
t = Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
print(t.argsort().numpy())
```


#### topk(k: 'int', dim: 'int' = -1, largest: 'bool' = True, sorted_: 'bool' = True) -> tuple[Tensor, Tensor]

Computes the top-k elements of the tensor along the specified `dim`.

Order of indices for equivalent elements is always preserved.

```python
t = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
print(t.numpy())
```


#### multinomial(num_samples: 'int' = 1, replacement: 'bool' = False) -> Tensor

Returns a tensor with `num_samples` indices sampled from a multinomial distribution weighted by `self`.

NOTE: `replacement=False` for `num_samples > 1` is not supported yet.

```python
Tensor.manual_seed(42)
t = Tensor([1, 2, 3, 4])
print(t.multinomial(20, replacement=True).numpy())
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


#### sequential(ll: 'list[Callable[[Tensor], Tensor]]') -> Tensor

Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

```python
t = Tensor([1, 2, 3])
print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
```


#### layernorm(axis: 'int | tuple[int, ...]' = -1, eps: 'float' = 1e-05) -> Tensor

Applies Layer Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1607.06450v1

```python
t = Tensor.randn(8, 10, 16) * 2 + 8
print(t.mean().item(), t.std().item())
```


#### batchnorm(weight: 'Tensor | None', bias: 'Tensor | None', mean: 'Tensor', invstd: 'Tensor', axis: 'int | tuple[int, ...]' = 1) -> Tensor

Applies Batch Normalization over a mini-batch of inputs.

- Paper: https://arxiv.org/abs/1502.03167

```python
t = Tensor.randn(8, 4, 16, 16) * 2 + 8
print(t.mean().item(), t.std().item())
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


#### one_hot(num_classes: 'int' = -1) -> Tensor

Converts `self` to a one-hot tensor.

`num_classes` defaults to -1, which means num_classes will be inferred as max(self) + 1.

```python
t = Tensor([0, 1, 3, 3, 4])
print(t.one_hot(5).numpy())
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


#### binary_crossentropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean') -> Tensor

Computes the binary cross-entropy loss between `self` and `Y`.

See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

```python
t = Tensor([0.1, 0.9, 0.2])
Y = Tensor([0, 1, 0])
print(t.binary_crossentropy(Y).item())
```


#### binary_crossentropy_logits(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', pos_weight: 'Tensor | None' = None) -> Tensor

Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.

See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

```python
t = Tensor([-1, 2, -3])
Y = Tensor([0, 1, 0])
print(t.binary_crossentropy_logits(Y).item())
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


#### cross_entropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', label_smoothing: 'float' = 0.0) -> Tensor

Computes the cross entropy loss between input logits and target.

NOTE: `self` are logits and `Y` are the target labels or class probabilities.

See: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

```python
t = Tensor([[-1, 2, -3], [1, -2, 3]])
Y = Tensor([1, 2])
print(t.cross_entropy(Y).item())
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


## Linear Algebra

#### qr() -> tuple[Tensor, Tensor]


#### svd(full_matrices=True) -> tuple[Tensor, Tensor, Tensor]

