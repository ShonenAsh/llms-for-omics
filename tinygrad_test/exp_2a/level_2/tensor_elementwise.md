Elementwise ops operate on a per element basis. They don't change the shape of the tensor.

## Unary Ops (math)

#### Tensor.logical_not() -> Tensor

Computes the logical NOT of the tensor element-wise.


#### Tensor.neg() -> Tensor

Negates the tensor element-wise.


#### Tensor.log() -> Tensor

Computes the natural logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm


#### Tensor.log2() -> Tensor

Computes the base-2 logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm


#### log10 *(resolution failed)*

#### Tensor.exp() -> Tensor

Computes the exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function


#### Tensor.exp2() -> Tensor

Computes the base-2 exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function


#### Tensor.sqrt() -> Tensor

Computes the square root of the tensor element-wise.


#### Tensor.rsqrt()

Computes the reciprocal of the square root of the tensor element-wise.


#### Tensor.sin() -> Tensor

Computes the sine of the tensor element-wise.


#### Tensor.cos() -> Tensor

Computes the cosine of the tensor element-wise.


#### Tensor.tan() -> Tensor

Computes the tangent of the tensor element-wise.


#### Tensor.asin() -> Tensor

Computes the inverse sine (arcsine) of the tensor element-wise.


#### Tensor.acos() -> Tensor

Computes the inverse cosine (arccosine) of the tensor element-wise.


#### Tensor.atan() -> Tensor

Computes the inverse tangent (arctan) of the tensor element-wise.


#### Tensor.trunc()

Truncates the tensor element-wise.


#### Tensor.ceil()

Rounds the tensor element-wise towards positive infinity.


#### Tensor.floor()

Rounds the tensor element-wise towards negative infinity.


#### Tensor.round() -> Tensor

Rounds the tensor element-wise with rounding half to even.


#### Tensor.isinf(detect_positive: bool = True, detect_negative: bool = True)

Checks the tensor element-wise to return True where the element is infinity, otherwise returns False


#### Tensor.isnan()

Checks the tensor element-wise to return True where the element is NaN, otherwise returns False


#### Tensor.isfinite()

Checks the tensor element-wise to return True where the element is finite, otherwise returns False


#### Tensor.lerp(end: 'Tensor', weight: 'Tensor | float') -> Tensor

Linearly interpolates between `self` and `end` by `weight`.


#### Tensor.square()

Squares the tensor element-wise.
Equivalent to `self*self`.


#### Tensor.clamp(min_=None, max_=None)

Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.


#### Tensor.clip(min_=None, max_=None)

Alias for `Tensor.clamp`.


#### Tensor.sign() -> Tensor

Returns the sign of the tensor element-wise.


#### Tensor.abs() -> Tensor

Computes the absolute value of the tensor element-wise.


#### Tensor.reciprocal() -> Tensor

Computes `1/x` element-wise.


## Unary Ops (activation)

#### Tensor.relu()

Applies the Rectified Linear Unit (ReLU) function element-wise.


#### Tensor.sigmoid()

Applies the Sigmoid function element-wise.

- Described: https://en.wikipedia.org/wiki/Sigmoid_function


#### Tensor.logsigmoid() -> Tensor

Applies the LogSigmoid function element-wise.

- See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html


#### Tensor.hardsigmoid(alpha: float = 0.16666666666666666, beta: float = 0.5)

Applies the Hardsigmoid function element-wise.
NOTE: default `alpha` and `beta` values are taken from torch

- See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html


#### Tensor.elu(alpha=1.0) -> Tensor

Applies the Exponential Linear Unit (ELU) function element-wise.

- Paper: https://arxiv.org/abs/1511.07289v5


#### Tensor.celu(alpha=1.0) -> Tensor

Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

- Paper: https://arxiv.org/abs/1704.07483


#### Tensor.selu(alpha=1.67326, gamma=1.0507) -> Tensor

Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

- Paper: https://arxiv.org/abs/1706.02515v5


#### Tensor.swish()

See `.silu()`

- Paper: https://arxiv.org/abs/1710.05941v1


#### Tensor.silu()

Applies the Sigmoid Linear Unit (SiLU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415


#### Tensor.relu6()

Applies the ReLU6 function element-wise.

- Paper: https://arxiv.org/abs/1704.04861v1


#### Tensor.hardswish()

Applies the Hardswish function element-wise.

- Paper: https://arxiv.org/abs/1905.02244v5


#### Tensor.tanh()

Applies the Hyperbolic Tangent (tanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh


#### Tensor.sinh() -> Tensor

Applies the Hyperbolic Sine (sinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh


#### Tensor.cosh() -> Tensor

Applies the Hyperbolic Cosine (cosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh


#### Tensor.atanh() -> Tensor

Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh


#### Tensor.asinh() -> Tensor

Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh


#### Tensor.acosh() -> Tensor

Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh


#### Tensor.hardtanh(min_val=-1, max_val=1)

Applies the Hardtanh function element-wise.


#### Tensor.erf() -> Tensor

Applies error function element-wise.

- Described: https://en.wikipedia.org/wiki/Error_function


#### Tensor.gelu()

Applies the Gaussian Error Linear Unit (GELU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415v5


#### Tensor.quick_gelu()

Applies the Sigmoid GELU approximation element-wise.


#### Tensor.leaky_relu(neg_slope=0.01)

Applies the Leaky ReLU function element-wise.


#### Tensor.mish() -> Tensor

Applies the Mish function element-wise.

- Paper: https://arxiv.org/abs/1908.08681v3


#### Tensor.softplus(beta=1.0) -> Tensor

Applies the Softplus function element-wise.


#### Tensor.softsign() -> Tensor

Applies the Softsign function element-wise.


## Elementwise Ops (broadcasted)

#### Tensor.add(x: Self | float | int | bool, reverse: bool = False)

Adds `self` and `x`.
Equivalent to `self + x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.


#### Tensor.sub(x: 'Tensor | ConstType', reverse=False) -> Tensor

Subtracts `x` from `self`.
Equivalent to `self - x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.


#### Tensor.mul(x: Self | float | int | bool, reverse: bool = False)

Multiplies `self` and `x`.
Equivalent to `self * x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.


#### Tensor.div(x: 'Tensor | ConstType', reverse=False, rounding_mode: "Literal['trunc', 'floor'] | None" = None) -> Tensor

Divides `self` by `x`.
Equivalent to `self / x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
`div` performs true division.


#### Tensor.idiv(x: Self | float | int | bool, reverse: bool = False)

Divides `self` by `x`.
Equivalent to `self // x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.
`idiv` performs integer division (truncate towards zero).


#### Tensor.mod(x: 'Tensor | ConstType', reverse=False) -> Tensor

Mod `self` by `x`.
Equivalent to `self % x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.


#### Tensor.bitwise_xor(x: Self | float | int | bool, reverse: bool = False)

Computes bitwise xor of `self` and `x`.
Equivalent to `self ^ x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.


#### Tensor.bitwise_and(x: Self | float | int | bool, reverse: bool = False)

Computes the bitwise AND of `self` and `x`.
Equivalent to `self & x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.


#### Tensor.bitwise_or(x: Self | float | int | bool, reverse: bool = False)

Computes the bitwise OR of `self` and `x`.
Equivalent to `self | x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.


#### Tensor.bitwise_not() -> Tensor

Computes the bitwise NOT of `self`.
Equivalent to `~self`.


#### Tensor.lshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self << x`.


#### Tensor.rshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self >> x`.


#### Tensor.pow(x: 'Tensor | ConstType', reverse=False) -> Tensor

Computes power of `self` with `x`.
Equivalent to `self ** x`.


#### Tensor.maximum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise maximum of `self` and `x`.


#### Tensor.minimum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise minimum of `self` and `x`.


#### Tensor.where(x: 'Tensor | ConstType | sint', y: 'Tensor | ConstType | sint') -> Tensor

Returns a tensor of elements selected from either `x` or `y`, depending on `self`.
`output_i = x_i if self_i else y_i`.


#### Tensor.copysign(other) -> Tensor

Returns a tensor of with the magnitude of `self` and the sign of `other`, elementwise.


#### Tensor.logaddexp(other) -> Tensor

Calculates (self.exp()+other.exp()).log(), elementwise.


## Casting Ops

#### Tensor.cast(dtype: 'DTypeLike') -> Tensor

Casts `self` to the given `dtype`.


#### Tensor.bitcast(dtype: 'DTypeLike') -> Tensor

Bitcasts `self` to the given `dtype` of the same itemsize.

`self` must not require a gradient.


#### Tensor.float() -> Tensor

Convenience method to cast `self` to a `float32` Tensor.


#### Tensor.half() -> Tensor

Convenience method to cast `self` to a `float16` Tensor.


#### Tensor.int() -> Tensor

Convenience method to cast `self` to a `int32` Tensor.


#### Tensor.bool() -> Tensor

Convenience method to cast `self` to a `bool` Tensor.


#### Tensor.bfloat16() -> Tensor


#### Tensor.double() -> Tensor


#### Tensor.long() -> Tensor


#### Tensor.short() -> Tensor

