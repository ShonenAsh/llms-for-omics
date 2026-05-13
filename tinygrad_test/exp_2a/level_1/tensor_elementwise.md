Elementwise ops operate on a per element basis. They don't change the shape of the tensor.

## Unary Ops (math)

#### Tensor.logical_not() -> Tensor


#### Tensor.neg() -> Tensor


#### Tensor.log() -> Tensor


#### Tensor.log2() -> Tensor


#### log10 *(resolution failed)*

#### Tensor.exp() -> Tensor


#### Tensor.exp2() -> Tensor


#### Tensor.sqrt() -> Tensor


#### Tensor.rsqrt()


#### Tensor.sin() -> Tensor


#### Tensor.cos() -> Tensor


#### Tensor.tan() -> Tensor


#### Tensor.asin() -> Tensor


#### Tensor.acos() -> Tensor


#### Tensor.atan() -> Tensor


#### Tensor.trunc()


#### Tensor.ceil()


#### Tensor.floor()


#### Tensor.round() -> Tensor


#### Tensor.isinf(detect_positive: bool = True, detect_negative: bool = True)


#### Tensor.isnan()


#### Tensor.isfinite()


#### Tensor.lerp(end: 'Tensor', weight: 'Tensor | float') -> Tensor


#### Tensor.square()


#### Tensor.clamp(min_=None, max_=None)


#### Tensor.clip(min_=None, max_=None)


#### Tensor.sign() -> Tensor


#### Tensor.abs() -> Tensor


#### Tensor.reciprocal() -> Tensor


## Unary Ops (activation)

#### Tensor.relu()


#### Tensor.sigmoid()


#### Tensor.logsigmoid() -> Tensor


#### Tensor.hardsigmoid(alpha: float = 0.16666666666666666, beta: float = 0.5)


#### Tensor.elu(alpha=1.0) -> Tensor


#### Tensor.celu(alpha=1.0) -> Tensor


#### Tensor.selu(alpha=1.67326, gamma=1.0507) -> Tensor


#### Tensor.swish()


#### Tensor.silu()


#### Tensor.relu6()


#### Tensor.hardswish()


#### Tensor.tanh()


#### Tensor.sinh() -> Tensor


#### Tensor.cosh() -> Tensor


#### Tensor.atanh() -> Tensor


#### Tensor.asinh() -> Tensor


#### Tensor.acosh() -> Tensor


#### Tensor.hardtanh(min_val=-1, max_val=1)


#### Tensor.erf() -> Tensor


#### Tensor.gelu()


#### Tensor.quick_gelu()


#### Tensor.leaky_relu(neg_slope=0.01)


#### Tensor.mish() -> Tensor


#### Tensor.softplus(beta=1.0) -> Tensor


#### Tensor.softsign() -> Tensor


## Elementwise Ops (broadcasted)

#### Tensor.add(x: Self | float | int | bool, reverse: bool = False)


#### Tensor.sub(x: 'Tensor | ConstType', reverse=False) -> Tensor


#### Tensor.mul(x: Self | float | int | bool, reverse: bool = False)


#### Tensor.div(x: 'Tensor | ConstType', reverse=False, rounding_mode: "Literal['trunc', 'floor'] | None" = None) -> Tensor


#### Tensor.idiv(x: Self | float | int | bool, reverse: bool = False)


#### Tensor.mod(x: 'Tensor | ConstType', reverse=False) -> Tensor


#### Tensor.bitwise_xor(x: Self | float | int | bool, reverse: bool = False)


#### Tensor.bitwise_and(x: Self | float | int | bool, reverse: bool = False)


#### Tensor.bitwise_or(x: Self | float | int | bool, reverse: bool = False)


#### Tensor.bitwise_not() -> Tensor


#### Tensor.lshift(x: 'Tensor | int', reverse=False) -> Tensor


#### Tensor.rshift(x: 'Tensor | int', reverse=False) -> Tensor


#### Tensor.pow(x: 'Tensor | ConstType', reverse=False) -> Tensor


#### Tensor.maximum(x: 'Tensor | ConstType') -> Tensor


#### Tensor.minimum(x: 'Tensor | ConstType') -> Tensor


#### Tensor.where(x: 'Tensor | ConstType | sint', y: 'Tensor | ConstType | sint') -> Tensor


#### Tensor.copysign(other) -> Tensor


#### Tensor.logaddexp(other) -> Tensor


## Casting Ops

#### Tensor.cast(dtype: 'DTypeLike') -> Tensor


#### Tensor.bitcast(dtype: 'DTypeLike') -> Tensor


#### Tensor.float() -> Tensor


#### Tensor.half() -> Tensor


#### Tensor.int() -> Tensor


#### Tensor.bool() -> Tensor


#### Tensor.bfloat16() -> Tensor


#### Tensor.double() -> Tensor


#### Tensor.long() -> Tensor


#### Tensor.short() -> Tensor

