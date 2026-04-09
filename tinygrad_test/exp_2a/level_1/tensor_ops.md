## Reduce

#### sum(axis: 'int | Sequence[int] | None' = None, keepdim=False, dtype: 'DTypeLike | None' = None) -> Tensor


#### prod(axis: 'int | Sequence[int] | None' = None, keepdim=False, dtype: 'DTypeLike | None' = None) -> Tensor


#### max(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### min(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### any(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### all(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### isclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> Tensor


#### allclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> bool


#### mean(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### var(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor


#### var_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]


#### std(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor


#### std_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]


#### softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor


#### log_softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor


#### logsumexp(axis=None, keepdim=False) -> Tensor


#### logcumsumexp(axis=0) -> Tensor


#### argmax(axis=None, keepdim=False) -> Tensor


#### argmin(axis=None, keepdim=False) -> Tensor


## Processing

#### avg_pool2d(kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, ceil_mode=False, count_include_pad=True) -> Tensor


#### max_pool2d(kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, ceil_mode=False, return_indices=False) -> Tensor | tuple[Tensor, Tensor]


#### max_unpool2d(indices: 'Tensor', kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, output_size=None)


#### conv2d(weight: 'Tensor', bias: 'Tensor | None' = None, groups=1, stride=1, dilation=1, padding: 'int | tuple[int, ...]' = 0, dtype: 'DTypeLike | None' = None) -> Tensor


#### conv_transpose2d(weight: 'Tensor', bias: 'Tensor | None' = None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor


#### dot(w: 'Tensor', dtype: 'DTypeLike | None' = None) -> Tensor


#### matmul(x: 'Tensor', reverse=False, dtype: 'DTypeLike | None' = None) -> Tensor


#### einsum(formula: 'str', *operands: 'Tensor | Sequence[Tensor]', dtype: 'DTypeLike | None' = None) -> Tensor


#### cumsum(axis: 'int' = 0) -> Tensor


#### cumprod(axis: 'int') -> Tensor


#### cummax(axis: 'int' = 0) -> Tensor


#### cummin *(resolution failed)*

#### triu(diagonal: 'int' = 0) -> Tensor


#### tril(diagonal: 'int' = 0) -> Tensor


#### interpolate(size: 'tuple[int, ...]', mode: 'str' = 'linear', align_corners: 'bool' = False) -> Tensor


#### scatter(dim: 'int', index: 'Tensor', src: 'Tensor | ConstType', reduce: "Literal['multiply', 'add'] | None" = None) -> Tensor


#### scatter_reduce(dim: 'int', index: 'Tensor', src: 'Tensor', reduce: "Literal['sum', 'prod', 'mean', 'amax', 'amin']", include_self: 'bool' = True) -> Tensor


#### masked_select(mask)


#### masked_fill(mask: 'Tensor', value: 'Tensor | ConstType') -> Tensor


#### nonzero *(resolution failed)*

#### sort(dim: 'int' = -1, descending: 'bool' = False) -> tuple[Tensor, Tensor]


#### argsort(dim: 'int' = -1, descending: 'bool' = False) -> Tensor


#### topk(k: 'int', dim: 'int' = -1, largest: 'bool' = True, sorted_: 'bool' = True) -> tuple[Tensor, Tensor]


#### multinomial(num_samples: 'int' = 1, replacement: 'bool' = False) -> Tensor


## Neural Network (functional)

#### linear(weight: 'Tensor', bias: 'Tensor | None' = None, dtype: 'DTypeLike | None' = None) -> Tensor


#### sequential(ll: 'list[Callable[[Tensor], Tensor]]') -> Tensor


#### layernorm(axis: 'int | tuple[int, ...]' = -1, eps: 'float' = 1e-05) -> Tensor


#### batchnorm(weight: 'Tensor | None', bias: 'Tensor | None', mean: 'Tensor', invstd: 'Tensor', axis: 'int | tuple[int, ...]' = 1) -> Tensor


#### dropout(p=0.5) -> Tensor


#### one_hot(num_classes: 'int' = -1) -> Tensor


#### scaled_dot_product_attention(key: 'Tensor', value: 'Tensor', attn_mask: 'Tensor | None' = None, dropout_p: 'float' = 0.0, is_causal: 'bool' = False, enable_gqa: 'bool' = False) -> Tensor


#### binary_crossentropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean') -> Tensor


#### binary_crossentropy_logits(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', pos_weight: 'Tensor | None' = None) -> Tensor


#### sparse_categorical_crossentropy(Y: 'Tensor', ignore_index: 'int' = -1, label_smoothing=0.0, reduction: 'ReductionStr' = 'mean') -> Tensor


#### cross_entropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', label_smoothing: 'float' = 0.0) -> Tensor


#### nll_loss(Y: 'Tensor', weight: 'Tensor | None' = None, ignore_index: 'int | None' = None, reduction: 'ReductionStr' = 'mean') -> Tensor


## Linear Algebra

#### qr() -> tuple[Tensor, Tensor]


#### svd(full_matrices=True) -> tuple[Tensor, Tensor, Tensor]

