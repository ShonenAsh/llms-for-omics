Elementwise ops operate on a per element basis. They don't change the shape of the tensor.

## Unary Ops (math)

#### logical_not() -> Tensor


#### neg() -> Tensor


#### log() -> Tensor


#### log2() -> Tensor


#### log10 *(resolution failed)*

#### exp() -> Tensor


#### exp2() -> Tensor


#### sqrt() -> Tensor


#### rsqrt()


#### sin() -> Tensor


#### cos() -> Tensor


#### tan() -> Tensor


#### asin() -> Tensor


#### acos() -> Tensor


#### atan() -> Tensor


#### trunc()


#### ceil()


#### floor()


#### round() -> Tensor


#### isinf(detect_positive: bool = True, detect_negative: bool = True)


#### isnan()


#### isfinite()


#### lerp(end: 'Tensor', weight: 'Tensor | float') -> Tensor


#### square()


#### clamp(min_=None, max_=None)


#### clip(min_=None, max_=None)


#### sign() -> Tensor


#### abs() -> Tensor


#### reciprocal() -> Tensor


## Unary Ops (activation)

#### relu()


#### sigmoid()


#### logsigmoid() -> Tensor


#### hardsigmoid(alpha: float = 0.16666666666666666, beta: float = 0.5)


#### elu(alpha=1.0) -> Tensor


#### celu(alpha=1.0) -> Tensor


#### selu(alpha=1.67326, gamma=1.0507) -> Tensor


#### swish()


#### silu()


#### relu6()


#### hardswish()


#### tanh()


#### sinh() -> Tensor


#### cosh() -> Tensor


#### atanh() -> Tensor


#### asinh() -> Tensor


#### acosh() -> Tensor


#### hardtanh(min_val=-1, max_val=1)


#### erf() -> Tensor


#### gelu()


#### quick_gelu()


#### leaky_relu(neg_slope=0.01)


#### mish() -> Tensor


#### softplus(beta=1.0) -> Tensor


#### softsign() -> Tensor


## Elementwise Ops (broadcasted)

#### add(x: Union[Self, float, int, bool], reverse: bool = False)


#### sub(x: 'Tensor | ConstType', reverse=False) -> Tensor


#### mul(x: Union[Self, float, int, bool], reverse: bool = False)


#### div(x: 'Tensor | ConstType', reverse=False, rounding_mode: "Literal['trunc', 'floor'] | None" = None) -> Tensor


#### idiv(x: Union[Self, float, int, bool], reverse: bool = False)


#### mod(x: 'Tensor | ConstType', reverse=False) -> Tensor


#### bitwise_xor(x: Union[Self, float, int, bool], reverse: bool = False)


#### bitwise_and(x: Union[Self, float, int, bool], reverse: bool = False)


#### bitwise_or(x: Union[Self, float, int, bool], reverse: bool = False)


#### bitwise_not() -> Tensor


#### lshift(x: 'Tensor | int', reverse=False) -> Tensor


#### rshift(x: 'Tensor | int', reverse=False) -> Tensor


#### pow(x: 'Tensor | ConstType', reverse=False) -> Tensor


#### maximum(x: 'Tensor | ConstType') -> Tensor


#### minimum(x: 'Tensor | ConstType') -> Tensor


#### where(x: 'Tensor | ConstType | sint', y: 'Tensor | ConstType | sint') -> Tensor


#### copysign(other) -> Tensor


#### logaddexp(other) -> Tensor


## Casting Ops

#### cast(dtype: 'DTypeLike') -> Tensor


#### bitcast(dtype: 'DTypeLike') -> Tensor


#### float() -> Tensor


#### half() -> Tensor


#### int() -> Tensor


#### bool() -> Tensor


#### bfloat16() -> Tensor


#### double() -> Tensor


#### long() -> Tensor


#### short() -> Tensor

