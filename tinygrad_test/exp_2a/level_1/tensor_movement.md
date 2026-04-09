## Movement (low level)

#### view(shape, *args) -> Self


#### reshape(shape, *args) -> Self


#### expand(shape, *args) -> Self


#### permute(order, *args) -> Self


#### flip(axis, *args) -> Self


#### shrink(arg: tuple[tuple['UOp | int', 'UOp | int'] | None, ...]) -> Self


#### pad(padding: 'Sequence[sint] | Sequence[tuple[sint, sint] | None]', mode: 'str' = 'constant', value: 'float' = 0.0) -> Tensor


## Movement (high level)

#### __getitem__(indices) -> Tensor


#### gather(dim: 'int', index: 'Tensor') -> Tensor


#### cat(*args: 'Tensor', dim: 'int' = 0) -> Tensor


#### stack(*args: 'Tensor', dim: 'int' = 0) -> Tensor


#### repeat(repeats, *args) -> Self


#### repeat_interleave(repeats: int, dim: int | None = None) -> Self


#### split(sizes: 'int | Sequence[int]', dim: 'int' = 0) -> tuple[Tensor, ...]


#### chunk(chunks: 'int', dim: 'int' = 0) -> list[Tensor]


#### unfold(dim: 'int', size: 'sint', step: 'int') -> Tensor


#### meshgrid(*args: 'Tensor', indexing: "Literal['ij', 'xy']" = 'ij') -> tuple[Tensor, ...]


#### squeeze(dim: int | None = None) -> Self


#### unsqueeze(dim: int) -> Self


#### T -> typing.Self


#### transpose(dim0=1, dim1=0) -> Self


#### flatten(start_dim=0, end_dim=-1) -> Self


#### unflatten(dim: int, sizes: tuple[int, ...]) -> Self


#### diag() -> Tensor


#### diagonal() -> Tensor


#### roll(shifts: 'int | tuple[int, ...]', dims: 'int | tuple[int, ...] | None' = None) -> Tensor


#### rearrange(formula: str, **sizes) -> Self

