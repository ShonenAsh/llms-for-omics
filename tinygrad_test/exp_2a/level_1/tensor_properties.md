## Basic

#### shape -> tuple[sint, ...]


#### dtype -> DType


#### device -> str | tuple[str, ...]


#### ndim -> <class 'int'>


#### numel() -> UOp | int


#### element_size() -> int


#### nbytes() -> int


#### is_floating_point() -> bool


#### size(dim: 'int | None' = None) -> sint | tuple[sint, ...]


## Data Access

#### data() -> memoryview


#### item() -> ConstType


#### tolist() -> Sequence[ConstType] | ConstType


#### numpy() -> 'numpy.ndarray'


## tinygrad ops

#### schedule_with_vars(*lst: 'Tensor') -> tuple[list[ExecItem], dict[str, int]]


#### schedule(*lst: 'Tensor') -> list[ExecItem]


#### realize(*lst: 'Tensor', do_update_stats=True) -> Tensor


#### replace(x: 'Tensor', allow_shape_mismatch=False) -> Tensor


#### assign(x) -> Tensor


#### detach() -> Tensor


#### clone() -> Tensor


#### to(device: 'str | tuple[str, ...] | None') -> Tensor


#### to_(device: 'str | tuple[str, ...] | None') -> Tensor


#### shard(devices: 'tuple[str, ...]', axis: 'int | None' = None) -> Tensor


#### shard_(devices: 'tuple[str, ...]', axis: 'int | None' = None) -> Tensor


#### contiguous(*args, **kwargs) -> Tensor


#### contiguous_backward() -> Tensor


## Gradient

#### gradient(*targets: 'Tensor', gradient: 'Tensor | None' = None, materialize_grads=False) -> list[Tensor]


#### backward(gradient: 'Tensor | None' = None) -> Tensor

