Elementwise ops operate on a per element basis. They don't change the shape of the tensor.

## Unary Ops (math)

#### logical_not() -> Tensor

Computes the logical NOT of the tensor element-wise.

```python
print(Tensor([False, True]).logical_not().numpy())
```


#### neg() -> Tensor

Negates the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
```


#### log() -> Tensor

Computes the natural logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm

```python
print(Tensor([1., 2., 4., 8.]).log().numpy())
```


#### log2() -> Tensor

Computes the base-2 logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm

```python
print(Tensor([1., 2., 4., 8.]).log2().numpy())
```


#### log10 *(resolution failed)*

#### exp() -> Tensor

Computes the exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function

```python
print(Tensor([0., 1., 2., 3.]).exp().numpy())
```


#### exp2() -> Tensor

Computes the base-2 exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function

```python
print(Tensor([0., 1., 2., 3.]).exp2().numpy())
```


#### sqrt() -> Tensor

Computes the square root of the tensor element-wise.

```python
print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
```


#### rsqrt()

Computes the reciprocal of the square root of the tensor element-wise.

```python
print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
```


#### sin() -> Tensor

Computes the sine of the tensor element-wise.

```python
print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
```


#### cos() -> Tensor

Computes the cosine of the tensor element-wise.

```python
print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
```


#### tan() -> Tensor

Computes the tangent of the tensor element-wise.

```python
print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
```


#### asin() -> Tensor

Computes the inverse sine (arcsine) of the tensor element-wise.

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).asin().numpy())
```


#### acos() -> Tensor

Computes the inverse cosine (arccosine) of the tensor element-wise.

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
```


#### atan() -> Tensor

Computes the inverse tangent (arctan) of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).atan().numpy())
```


#### trunc()

Truncates the tensor element-wise.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
```


#### ceil()

Rounds the tensor element-wise towards positive infinity.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).ceil().numpy())
```


#### floor()

Rounds the tensor element-wise towards negative infinity.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).floor().numpy())
```


#### round() -> Tensor

Rounds the tensor element-wise with rounding half to even.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).round().numpy())
```


#### isinf(detect_positive: bool = True, detect_negative: bool = True)

Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
```


#### isnan()

Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
```


#### isfinite()

Checks the tensor element-wise to return True where the element is finite, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite().numpy())
```


#### lerp(end: 'Tensor', weight: 'Tensor | float') -> Tensor

Linearly interpolates between `self` and `end` by `weight`.

```python
print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
```


#### square()

Squares the tensor element-wise.
Equivalent to `self*self`.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
```


#### clamp(min_=None, max_=None)

Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
```


#### clip(min_=None, max_=None)

Alias for `Tensor.clamp`.


#### sign() -> Tensor

Returns the sign of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
```


#### abs() -> Tensor

Computes the absolute value of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
```


#### reciprocal() -> Tensor

Computes `1/x` element-wise.

```python
print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
```


## Unary Ops (activation)

#### relu()

Applies the Rectified Linear Unit (ReLU) function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
```


#### sigmoid()

Applies the Sigmoid function element-wise.

- Described: https://en.wikipedia.org/wiki/Sigmoid_function

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
```


#### logsigmoid() -> Tensor

Applies the LogSigmoid function element-wise.

- See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).logsigmoid().numpy())
```


#### hardsigmoid(alpha: float = 0.16666666666666666, beta: float = 0.5)

Applies the Hardsigmoid function element-wise.
NOTE: default `alpha` and `beta` values are taken from torch

- See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
```


#### elu(alpha=1.0) -> Tensor

Applies the Exponential Linear Unit (ELU) function element-wise.

- Paper: https://arxiv.org/abs/1511.07289v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
```


#### celu(alpha=1.0) -> Tensor

Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

- Paper: https://arxiv.org/abs/1704.07483

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
```


#### selu(alpha=1.67326, gamma=1.0507) -> Tensor

Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

- Paper: https://arxiv.org/abs/1706.02515v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
```


#### swish()

See `.silu()`

- Paper: https://arxiv.org/abs/1710.05941v1

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
```


#### silu()

Applies the Sigmoid Linear Unit (SiLU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
```


#### relu6()

Applies the ReLU6 function element-wise.

- Paper: https://arxiv.org/abs/1704.04861v1

```python
print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
```


#### hardswish()

Applies the Hardswish function element-wise.

- Paper: https://arxiv.org/abs/1905.02244v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
```


#### tanh()

Applies the Hyperbolic Tangent (tanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
```


#### sinh() -> Tensor

Applies the Hyperbolic Sine (sinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
```


#### cosh() -> Tensor

Applies the Hyperbolic Cosine (cosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
```


#### atanh() -> Tensor

Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
```


#### asinh() -> Tensor

Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
```


#### acosh() -> Tensor

Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
```


#### hardtanh(min_val=-1, max_val=1)

Applies the Hardtanh function element-wise.

```python
print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
```


#### erf() -> Tensor

Applies error function element-wise.

- Described: https://en.wikipedia.org/wiki/Error_function

```python
print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).erf().numpy())
```


#### gelu()

Applies the Gaussian Error Linear Unit (GELU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
```


#### quick_gelu()

Applies the Sigmoid GELU approximation element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
```


#### leaky_relu(neg_slope=0.01)

Applies the Leaky ReLU function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu().numpy())
```


#### mish() -> Tensor

Applies the Mish function element-wise.

- Paper: https://arxiv.org/abs/1908.08681v3

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
```


#### softplus(beta=1.0) -> Tensor

Applies the Softplus function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
```


#### softsign() -> Tensor

Applies the Softsign function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
```


## Elementwise Ops (broadcasted)

#### add(x: Union[Self, float, int, bool], reverse: bool = False)

Adds `self` and `x`.
Equivalent to `self + x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```


#### sub(x: 'Tensor | ConstType', reverse=False) -> Tensor

Subtracts `x` from `self`.
Equivalent to `self - x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```


#### mul(x: Union[Self, float, int, bool], reverse: bool = False)

Multiplies `self` and `x`.
Equivalent to `self * x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```


#### div(x: 'Tensor | ConstType', reverse=False, rounding_mode: "Literal['trunc', 'floor'] | None" = None) -> Tensor

Divides `self` by `x`.
Equivalent to `self / x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
`div` performs true division.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```


#### idiv(x: Union[Self, float, int, bool], reverse: bool = False)

Divides `self` by `x`.
Equivalent to `self // x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.
`idiv` performs integer division (truncate towards zero).

```python
print(Tensor([-4, 7, 5, 4, -7, 8]).idiv(Tensor([2, -3, 8, -2, 3, 5])).numpy())
```


#### mod(x: 'Tensor | ConstType', reverse=False) -> Tensor

Mod `self` by `x`.
Equivalent to `self % x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.

```python
print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
```


#### bitwise_xor(x: Union[Self, float, int, bool], reverse: bool = False)

Computes bitwise xor of `self` and `x`.
Equivalent to `self ^ x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([-1, -2, 3]).bitwise_xor(Tensor([1, 0, 3])).numpy())
```


#### bitwise_and(x: Union[Self, float, int, bool], reverse: bool = False)

Computes the bitwise AND of `self` and `x`.
Equivalent to `self & x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
```


#### bitwise_or(x: Union[Self, float, int, bool], reverse: bool = False)

Computes the bitwise OR of `self` and `x`.
Equivalent to `self | x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
```


#### bitwise_not() -> Tensor

Computes the bitwise NOT of `self`.
Equivalent to `~self`.

```python
print(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
```


#### lshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self << x`.

```python
print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
```


#### rshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self >> x`.

```python
print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
```


#### pow(x: 'Tensor | ConstType', reverse=False) -> Tensor

Computes power of `self` with `x`.
Equivalent to `self ** x`.

```python
print(Tensor([-1, 2, 3]).pow(2.0).numpy())
```


#### maximum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise maximum of `self` and `x`.

```python
print(Tensor([-1, 2, 3]).maximum(1).numpy())
```


#### minimum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise minimum of `self` and `x`.

```python
print(Tensor([-1, 2, 3]).minimum(1).numpy())
```


#### where(x: 'Tensor | ConstType | sint', y: 'Tensor | ConstType | sint') -> Tensor

Returns a tensor of elements selected from either `x` or `y`, depending on `self`.
`output_i = x_i if self_i else y_i`.

```python
cond = Tensor([[True, True, False], [True, False, False]])
print(cond.where(1, 3).numpy())
```


#### copysign(other) -> Tensor

Returns a tensor of with the magnitude of `self` and the sign of `other`, elementwise.


#### logaddexp(other) -> Tensor

Calculates (self.exp()+other.exp()).log(), elementwise.


## Casting Ops

#### cast(dtype: 'DTypeLike') -> Tensor

Casts `self` to the given `dtype`.

```python
t = Tensor([-1, 2.5, 3], dtype=dtypes.float)
print(t.dtype, t.numpy())
```


#### bitcast(dtype: 'DTypeLike') -> Tensor

Bitcasts `self` to the given `dtype` of the same itemsize.

`self` must not require a gradient.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```


#### float() -> Tensor

Convenience method to cast `self` to a `float32` Tensor.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```


#### half() -> Tensor

Convenience method to cast `self` to a `float16` Tensor.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```


#### int() -> Tensor

Convenience method to cast `self` to a `int32` Tensor.

```python
t = Tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
print(t.dtype, t.numpy())
```


#### bool() -> Tensor

Convenience method to cast `self` to a `bool` Tensor.

```python
t = Tensor([-1, 0, 1])
print(t.dtype, t.numpy())
```


#### bfloat16() -> Tensor


#### double() -> Tensor


#### long() -> Tensor


#### short() -> Tensor

