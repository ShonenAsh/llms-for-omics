## Creation (basic)

#### empty(*shape, device: 'str | tuple[str, ...] | None' = None, dtype: 'DTypeLike | None' = None, **kwargs) -> Tensor


#### zeros(*shape, **kwargs) -> Tensor


#### ones(*shape, **kwargs) -> Tensor


#### full(shape: 'tuple[sint, ...]', fill_value: 'ConstType', **kwargs) -> Tensor


#### arange(start, stop=None, step=1, **kwargs) -> Tensor


#### linspace(start: 'int | float', stop: 'int | float', steps: 'int', **kwargs) -> Tensor


#### eye(n: 'int', m: 'int | None' = None, dtype=None, device=None, requires_grad: 'bool | None' = None) -> Tensor


#### full_like(fill_value: 'ConstType', **kwargs) -> Tensor


#### zeros_like(**kwargs) -> Tensor


#### ones_like(**kwargs) -> Tensor


## Creation (external)

#### from_blob(ptr: 'int', shape: 'tuple[int, ...]', **kwargs) -> Tensor


#### from_url(url: 'str', gunzip: 'bool' = False, **kwargs) -> Tensor


## Creation (random)

#### manual_seed(seed=0) -> None


#### rand(*shape, device: 'str | None' = None, dtype: 'DTypeLike | None' = None, contiguous: 'bool' = True, **kwargs) -> Tensor


#### rand_like(**kwargs) -> Tensor


#### randn(*shape, dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### randn_like(dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### randint(*shape, low=0, high=10, dtype=dtypes.int, **kwargs) -> Tensor


#### randperm(n: 'int', device=None, dtype=dtypes.int, **kwargs) -> Tensor


#### normal(*shape, mean=0.0, std=1.0, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### uniform(*shape, low=0.0, high=1.0, dtype: 'DTypeLike | None' = None, requires_grad: 'bool | None' = None, **kwargs) -> Tensor


#### scaled_uniform(*shape, **kwargs) -> Tensor


#### glorot_uniform(*shape, **kwargs) -> Tensor


#### kaiming_uniform(*shape, a: 'float' = 0.01, **kwargs) -> Tensor


#### kaiming_normal(*shape, a: 'float' = 0.01, **kwargs) -> Tensor

