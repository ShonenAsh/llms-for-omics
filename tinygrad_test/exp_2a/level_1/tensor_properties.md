## Basic

#### Tensor.shape -> tuple[sint, ...]


#### Tensor.dtype -> DType


#### Tensor.device -> str | tuple[str, ...]


#### Tensor.ndim -> <class 'int'>


#### Tensor.numel() -> UOp | int


#### Tensor.element_size() -> int


#### Tensor.nbytes() -> int


#### Tensor.is_floating_point() -> bool


#### Tensor.size(dim: 'int | None' = None) -> sint | tuple[sint, ...]


## Data Access

#### Tensor.data() -> memoryview


#### Tensor.item() -> ConstType


#### Tensor.tolist() -> Sequence[ConstType] | ConstType


#### Tensor.numpy() -> 'numpy.ndarray'


## tinygrad ops

#### Tensor.schedule_with_vars(*lst: 'Tensor') -> tuple[list[ExecItem], dict[str, int]]


#### Tensor.schedule(*lst: 'Tensor') -> list[ExecItem]


#### Tensor.realize(*lst: 'Tensor', do_update_stats=True) -> Tensor


#### Tensor.replace(x: 'Tensor', allow_shape_mismatch=False) -> Tensor


#### Tensor.assign(x) -> Tensor


#### Tensor.detach() -> Tensor


#### Tensor.clone() -> Tensor


#### Tensor.to(device: 'str | tuple[str, ...] | None') -> Tensor


#### Tensor.to_(device: 'str | tuple[str, ...] | None') -> Tensor


#### Tensor.shard(devices: 'tuple[str, ...]', axis: 'int | None' = None) -> Tensor


#### Tensor.shard_(devices: 'tuple[str, ...]', axis: 'int | None' = None) -> Tensor


#### Tensor.contiguous(*args, **kwargs) -> Tensor


#### Tensor.contiguous_backward() -> Tensor


## Gradient

#### Tensor.gradient(*targets: 'Tensor', gradient: 'Tensor | None' = None, materialize_grads=False) -> list[Tensor]


#### Tensor.backward(gradient: 'Tensor | None' = None) -> Tensor

