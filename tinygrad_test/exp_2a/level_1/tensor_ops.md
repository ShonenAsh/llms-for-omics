## Reduce

#### Tensor.sum(axis: 'int | Sequence[int] | None' = None, keepdim=False, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.prod(axis: 'int | Sequence[int] | None' = None, keepdim=False, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.max(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### Tensor.min(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### Tensor.any(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### Tensor.all(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### Tensor.isclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> Tensor


#### Tensor.allclose(other: 'Tensor', rtol: 'float' = 1e-05, atol: 'float' = 1e-08, equal_nan=False) -> bool


#### Tensor.mean(axis: 'int | Sequence[int] | None' = None, keepdim=False) -> Tensor


#### Tensor.var(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor


#### Tensor.var_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]


#### Tensor.std(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> Tensor


#### Tensor.std_mean(axis: 'int | Sequence[int] | None' = None, keepdim=False, correction=1) -> tuple[Tensor, Tensor]


#### Tensor.softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.log_softmax(axis=-1, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.logsumexp(axis=None, keepdim=False) -> Tensor


#### Tensor.logcumsumexp(axis=0) -> Tensor


#### Tensor.argmax(axis=None, keepdim=False) -> Tensor


#### Tensor.argmin(axis=None, keepdim=False) -> Tensor


## Processing

#### Tensor.avg_pool2d(kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, ceil_mode=False, count_include_pad=True) -> Tensor


#### Tensor.max_pool2d(kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, ceil_mode=False, return_indices=False) -> Tensor | tuple[Tensor, Tensor]


#### Tensor.max_unpool2d(indices: 'Tensor', kernel_size: 'tuple[int, ...]' = (2, 2), stride=None, dilation=1, padding: 'int | tuple[int, ...]' = 0, output_size=None)


#### Tensor.conv2d(weight: 'Tensor', bias: 'Tensor | None' = None, groups=1, stride=1, dilation=1, padding: 'int | tuple[int, ...]' = 0, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.conv_transpose2d(weight: 'Tensor', bias: 'Tensor | None' = None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor


#### Tensor.dot(w: 'Tensor', dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.matmul(x: 'Tensor', reverse=False, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.einsum(formula: 'str', *operands: 'Tensor | Sequence[Tensor]', dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.cumsum(axis: 'int' = 0) -> Tensor


#### Tensor.cumprod(axis: 'int') -> Tensor


#### Tensor.cummax(axis: 'int' = 0) -> Tensor


#### cummin *(resolution failed)*

#### Tensor.triu(diagonal: 'int' = 0) -> Tensor


#### Tensor.tril(diagonal: 'int' = 0) -> Tensor


#### Tensor.interpolate(size: 'tuple[int, ...]', mode: 'str' = 'linear', align_corners: 'bool' = False) -> Tensor


#### Tensor.scatter(dim: 'int', index: 'Tensor', src: 'Tensor | ConstType', reduce: "Literal['multiply', 'add'] | None" = None) -> Tensor


#### Tensor.scatter_reduce(dim: 'int', index: 'Tensor', src: 'Tensor', reduce: "Literal['sum', 'prod', 'mean', 'amax', 'amin']", include_self: 'bool' = True) -> Tensor


#### Tensor.masked_select(mask)


#### Tensor.masked_fill(mask: 'Tensor', value: 'Tensor | ConstType') -> Tensor


#### nonzero *(resolution failed)*

#### Tensor.sort(dim: 'int' = -1, descending: 'bool' = False) -> tuple[Tensor, Tensor]


#### Tensor.argsort(dim: 'int' = -1, descending: 'bool' = False) -> Tensor


#### Tensor.topk(k: 'int', dim: 'int' = -1, largest: 'bool' = True, sorted_: 'bool' = True) -> tuple[Tensor, Tensor]


#### Tensor.multinomial(num_samples: 'int' = 1, replacement: 'bool' = False) -> Tensor


## Neural Network (functional)

#### Tensor.linear(weight: 'Tensor', bias: 'Tensor | None' = None, dtype: 'DTypeLike | None' = None) -> Tensor


#### Tensor.sequential(ll: 'list[Callable[[Tensor], Tensor]]') -> Tensor


#### Tensor.layernorm(axis: 'int | tuple[int, ...]' = -1, eps: 'float' = 1e-05) -> Tensor


#### Tensor.batchnorm(weight: 'Tensor | None', bias: 'Tensor | None', mean: 'Tensor', invstd: 'Tensor', axis: 'int | tuple[int, ...]' = 1) -> Tensor


#### Tensor.dropout(p=0.5) -> Tensor


#### Tensor.one_hot(num_classes: 'int' = -1) -> Tensor


#### Tensor.scaled_dot_product_attention(key: 'Tensor', value: 'Tensor', attn_mask: 'Tensor | None' = None, dropout_p: 'float' = 0.0, is_causal: 'bool' = False, enable_gqa: 'bool' = False) -> Tensor


#### Tensor.binary_crossentropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean') -> Tensor


#### Tensor.binary_crossentropy_logits(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', pos_weight: 'Tensor | None' = None) -> Tensor


#### Tensor.sparse_categorical_crossentropy(Y: 'Tensor', ignore_index: 'int' = -1, label_smoothing=0.0, reduction: 'ReductionStr' = 'mean') -> Tensor


#### Tensor.cross_entropy(Y: 'Tensor', reduction: 'ReductionStr' = 'mean', label_smoothing: 'float' = 0.0) -> Tensor


#### Tensor.nll_loss(Y: 'Tensor', weight: 'Tensor | None' = None, ignore_index: 'int | None' = None, reduction: 'ReductionStr' = 'mean') -> Tensor


## Linear Algebra

#### Tensor.qr() -> tuple[Tensor, Tensor]


#### Tensor.svd(full_matrices=True) -> tuple[Tensor, Tensor, Tensor]

