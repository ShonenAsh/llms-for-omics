## Movement (low level)

#### Tensor.view(shape, *args) -> Self


#### Tensor.reshape(shape, *args) -> Self


#### Tensor.expand(shape, *args) -> Self


#### Tensor.permute(order, *args) -> Self


#### Tensor.flip(axis, *args) -> Self


#### Tensor.shrink(arg: tuple[tuple['UOp | int', 'UOp | int'] | None, ...]) -> Self


#### Tensor.pad(padding: 'Sequence[sint] | Sequence[tuple[sint, sint] | None]', mode: 'str' = 'constant', value: 'float' = 0.0) -> Tensor


## Movement (high level)

#### Tensor.__getitem__(indices) -> Tensor


#### Tensor.gather(dim: 'int', index: 'Tensor') -> Tensor


#### Tensor.cat(*args: 'Tensor', dim: 'int' = 0) -> Tensor


#### Tensor.stack(*args: 'Tensor', dim: 'int' = 0) -> Tensor


#### Tensor.repeat(repeats, *args) -> Self


#### Tensor.repeat_interleave(repeats: int, dim: int | None = None) -> Self


#### Tensor.split(sizes: 'int | Sequence[int]', dim: 'int' = 0) -> tuple[Tensor, ...]


#### Tensor.chunk(chunks: 'int', dim: 'int' = 0) -> list[Tensor]


#### Tensor.unfold(dim: 'int', size: 'sint', step: 'int') -> Tensor


#### Tensor.meshgrid(*args: 'Tensor', indexing: "Literal['ij', 'xy']" = 'ij') -> tuple[Tensor, ...]


#### Tensor.squeeze(dim: int | None = None) -> Self


#### Tensor.unsqueeze(dim: int) -> Self


#### Tensor.T -> typing.Self


#### Tensor.transpose(dim0=1, dim1=0) -> Self


#### Tensor.flatten(start_dim=0, end_dim=-1) -> Self


#### Tensor.unflatten(dim: int, sizes: tuple[int, ...]) -> Self


#### Tensor.diag() -> Tensor


#### Tensor.diagonal() -> Tensor


#### Tensor.roll(shifts: 'int | tuple[int, ...]', dims: 'int | tuple[int, ...] | None' = None) -> Tensor


#### Tensor.rearrange(formula: str, **sizes) -> Self

