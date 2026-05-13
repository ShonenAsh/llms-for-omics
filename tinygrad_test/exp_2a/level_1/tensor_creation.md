## Creation (basic)

#### Tensor.empty(*shape, device: 'str | tuple[str, ...] | None' = None, dtype: 'DTypeLike | None' = None, **kwargs) -> Tensor


#### Tensor.zeros(*shape, **kwargs) -> Tensor


#### Tensor.ones(*shape, **kwargs) -> Tensor


#### Tensor.full(shape: 'tuple[sint, ...]', fill_value: 'ConstType', **kwargs) -> Tensor


#### Tensor.arange(start, stop=None, step=1, **kwargs) -> Tensor


#### Tensor.linspace(start: 'int | float', stop: 'int | float', steps: 'int', **kwargs) -> Tensor


#### Tensor.eye(n: 'int', m: 'int | None' = None, dtype=None, device=None, requires_grad: 'bool | None' = None) -> Tensor


#### Tensor.full_like(fill_value: 'ConstType', **kwargs) -> Tensor


#### Tensor.zeros_like(**kwargs) -> Tensor


#### Tensor.ones_like(**kwargs) -> Tensor


## Creation (external)

#### Tensor.from_blob(ptr: 'int', shape: 'tuple[int, ...]', **kwargs) -> Tensor


#### Tensor.from_url(url: 'str', gunzip: 'bool' = False, **kwargs) -> Tensor


## Creation (random)

#### Tensor.manual_seed(seed=0) -> None


#### Tensor.rand(*shape, device: 'str | None' = None, dtype: 'DTypeLike | None' = None, contiguous: 'bool' = True, **kwargs) -> Tensor


#### Tensor.rand_like(**kwargs) -> Tensor


#### Tensor.randn(*shape, dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### Tensor.randn_like(dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### Tensor.randint(*shape, low=0, high=10, dtype=dtypes.int, **kwargs) -> Tensor


#### Tensor.randperm(n: 'int', device=None, dtype=dtypes.int, **kwargs) -> Tensor


#### Tensor.normal(*shape, mean=0.0, std=1.0, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### Tensor.uniform(*shape, low=0.0, high=1.0, dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### Tensor.scaled_uniform(*shape, **kwargs) -> Tensor


#### Tensor.glorot_uniform(*shape, **kwargs) -> Tensor


#### Tensor.kaiming_uniform(*shape, a: 'float' = 0.01, **kwargs) -> Tensor


#### Tensor.kaiming_normal(*shape, a: 'float' = 0.01, **kwargs) -> Tensor

