Elementwise ops operate on a per element basis. They don't change the shape of the tensor.

## Unary Ops (math)

#### logical_not() -> Tensor

Computes the logical NOT of the tensor element-wise.

```python
print(Tensor([False, True]).logical_not().numpy())
```

```
[ True False]
```


#### neg() -> Tensor

Negates the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
```

```
[ 3.  2.  1. -0. -1. -2. -3.]
```


#### log() -> Tensor

Computes the natural logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm

```python
print(Tensor([1., 2., 4., 8.]).log().numpy())
```

```
[0.     0.6931 1.3863 2.0794]
```


#### log2() -> Tensor

Computes the base-2 logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm

```python
print(Tensor([1., 2., 4., 8.]).log2().numpy())
```

```
[0. 1. 2. 3.]
```


#### log10 *(resolution failed)*

#### exp() -> Tensor

Computes the exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function

```python
print(Tensor([0., 1., 2., 3.]).exp().numpy())
```

```
[ 1.      2.7183  7.3891 20.0855]
```


#### exp2() -> Tensor

Computes the base-2 exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function

```python
print(Tensor([0., 1., 2., 3.]).exp2().numpy())
```

```
[1. 2. 4. 8.]
```


#### sqrt() -> Tensor

Computes the square root of the tensor element-wise.

```python
print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
```

```
[1.     1.4142 1.7321 2.    ]
```


#### rsqrt()

Computes the reciprocal of the square root of the tensor element-wise.

```python
print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
```

```
[1.     0.7071 0.5774 0.5   ]
```


#### sin() -> Tensor

Computes the sine of the tensor element-wise.

```python
print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
```

```
[ 0.  1. -0. -1.  0.]
```


#### cos() -> Tensor

Computes the cosine of the tensor element-wise.

```python
print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
```

```
[ 1.0000e+00  0.0000e+00 -1.0000e+00 -2.3842e-07  1.0000e+00]
```


#### tan() -> Tensor

Computes the tangent of the tensor element-wise.

```python
print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
```

```
[ 0.  1. inf -1.  0.]
```


#### asin() -> Tensor

Computes the inverse sine (arcsine) of the tensor element-wise.

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).asin().numpy())
```

```
[-1.1198 -0.6435 -0.3047  0.      0.3047  0.6435  1.1198]
```


#### acos() -> Tensor

Computes the inverse cosine (arccosine) of the tensor element-wise.

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
```

```
[2.6906 2.2143 1.8755 1.5708 1.2661 0.9273 0.451 ]
```


#### atan() -> Tensor

Computes the inverse tangent (arctan) of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).atan().numpy())
```

```
[-1.249  -1.1071 -0.7854  0.      0.7854  1.1071  1.249 ]
```


#### trunc()

Truncates the tensor element-wise.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
```

```
[-3. -2. -1. -0.  0.  1.  2.  3.]
```


#### ceil()

Rounds the tensor element-wise towards positive infinity.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).ceil().numpy())
```

```
[-3. -2. -1. -0.  1.  2.  3.  4.]
```


#### floor()

Rounds the tensor element-wise towards negative infinity.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).floor().numpy())
```

```
[-4. -3. -2. -1.  0.  1.  2.  3.]
```


#### round() -> Tensor

Rounds the tensor element-wise with rounding half to even.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).round().numpy())
```

```
[-4. -2. -2.  0.  0.  2.  2.  4.]
```


#### isinf(detect_positive: bool = True, detect_negative: bool = True)

Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
```

```
[False  True False  True False]
```


#### isnan()

Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
```

```
[False False False False  True]
```


#### isfinite()

Checks the tensor element-wise to return True where the element is finite, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite().numpy())
```

```
[ True False  True False False]
```


#### lerp(end: 'Tensor', weight: 'Tensor | float') -> Tensor

Linearly interpolates between `self` and `end` by `weight`.

```python
print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
```

```
[2.5 3.5 4.5]
```


#### square()

Squares the tensor element-wise.
Equivalent to `self*self`.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
```

```
[9. 4. 1. 0. 1. 4. 9.]
```


#### clamp(min_=None, max_=None)

Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
```

```
[-1. -1. -1.  0.  1.  1.  1.]
```


#### clip(min_=None, max_=None)

Alias for `Tensor.clamp`.


#### sign() -> Tensor

Returns the sign of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
```

```
[-1. -1. -1.  0.  1.  1.  1.]
```


#### abs() -> Tensor

Computes the absolute value of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
```

```
[3. 2. 1. 0. 1. 2. 3.]
```


#### reciprocal() -> Tensor

Computes `1/x` element-wise.

```python
print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
```

```
[1.     0.5    0.3333 0.25  ]
```


## Unary Ops (activation)

#### relu()

Applies the Rectified Linear Unit (ReLU) function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
```

```
[0. 0. 0. 0. 1. 2. 3.]
```


#### sigmoid()

Applies the Sigmoid function element-wise.

- Described: https://en.wikipedia.org/wiki/Sigmoid_function

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
```

```
[0.0474 0.1192 0.2689 0.5    0.7311 0.8808 0.9526]
```


#### logsigmoid() -> Tensor

Applies the LogSigmoid function element-wise.

- See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).logsigmoid().numpy())
```

```
[-3.0486 -2.1269 -1.3133 -0.6931 -0.3133 -0.1269 -0.0486]
```


#### hardsigmoid(alpha: float = 0.16666666666666666, beta: float = 0.5)

Applies the Hardsigmoid function element-wise.
NOTE: default `alpha` and `beta` values are taken from torch

- See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
```

```
[0.     0.1667 0.3333 0.5    0.6667 0.8333 1.    ]
```


#### elu(alpha=1.0) -> Tensor

Applies the Exponential Linear Unit (ELU) function element-wise.

- Paper: https://arxiv.org/abs/1511.07289v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
```

```
[-0.9502 -0.8647 -0.6321  0.      1.      2.      3.    ]
```


#### celu(alpha=1.0) -> Tensor

Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

- Paper: https://arxiv.org/abs/1704.07483

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
```

```
[-0.9502 -0.8647 -0.6321  0.      1.      2.      3.    ]
```


#### selu(alpha=1.67326, gamma=1.0507) -> Tensor

Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

- Paper: https://arxiv.org/abs/1706.02515v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
```

```
[-1.6706 -1.5202 -1.1113  0.      1.0507  2.1014  3.1521]
```


#### swish()

See `.silu()`

- Paper: https://arxiv.org/abs/1710.05941v1

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
```

```
[-0.1423 -0.2384 -0.2689  0.      0.7311  1.7616  2.8577]
```


#### silu()

Applies the Sigmoid Linear Unit (SiLU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
```

```
[-0.1423 -0.2384 -0.2689  0.      0.7311  1.7616  2.8577]
```


#### relu6()

Applies the ReLU6 function element-wise.

- Paper: https://arxiv.org/abs/1704.04861v1

```python
print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
```

```
[0. 0. 0. 0. 3. 6. 6.]
```


#### hardswish()

Applies the Hardswish function element-wise.

- Paper: https://arxiv.org/abs/1905.02244v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
```

```
[-0.     -0.3333 -0.3333  0.      0.6667  1.6667  3.    ]
```


#### tanh()

Applies the Hyperbolic Tangent (tanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
```

```
[-0.9951 -0.964  -0.7616  0.      0.7616  0.964   0.9951]
```


#### sinh() -> Tensor

Applies the Hyperbolic Sine (sinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
```

```
[-10.0179  -3.6269  -1.1752   0.       1.1752   3.6269  10.0179]
```


#### cosh() -> Tensor

Applies the Hyperbolic Cosine (cosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
```

```
[10.0677  3.7622  1.5431  1.      1.5431  3.7622 10.0677]
```


#### atanh() -> Tensor

Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
```

```
[-1.4722 -0.6931 -0.3095  0.      0.3095  0.6931  1.4722]
```


#### asinh() -> Tensor

Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
```

```
[-1.8184 -1.4436 -0.8814  0.      0.8814  1.4436  1.8184]
```


#### acosh() -> Tensor

Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
```

```
[   nan    nan    nan    nan 0.     1.317  1.7627]
```


#### hardtanh(min_val=-1, max_val=1)

Applies the Hardtanh function element-wise.

```python
print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
```

```
[-1.  -1.  -0.5  0.   0.5  1.   1. ]
```


#### erf() -> Tensor

Applies error function element-wise.

- Described: https://en.wikipedia.org/wiki/Error_function

```python
print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).erf().numpy())
```

```
[-0.9661 -0.8427 -0.5205  0.      0.5205  0.8427  0.9661]
```


#### gelu()

Applies the Gaussian Error Linear Unit (GELU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
```

```
[-0.0036 -0.0454 -0.1588  0.      0.8412  1.9546  2.9964]
```


#### quick_gelu()

Applies the Sigmoid GELU approximation element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
```

```
[-0.0181 -0.0643 -0.1542  0.      0.8458  1.9357  2.9819]
```


#### leaky_relu(neg_slope=0.01)

Applies the Leaky ReLU function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu().numpy())
```

```
[-0.03 -0.02 -0.01  0.    1.    2.    3.  ]
```

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu(neg_slope=0.42).numpy())
```

```
[-1.26 -0.84 -0.42  0.    1.    2.    3.  ]
```


#### mish() -> Tensor

Applies the Mish function element-wise.

- Paper: https://arxiv.org/abs/1908.08681v3

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
```

```
[-0.1456 -0.2525 -0.3034  0.      0.8651  1.944   2.9865]
```


#### softplus(beta=1.0) -> Tensor

Applies the Softplus function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
```

```
[0.0486 0.1269 0.3133 0.6931 1.3133 2.1269 3.0486]
```


#### softsign() -> Tensor

Applies the Softsign function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
```

```
[-0.75   -0.6667 -0.5     0.      0.5     0.6667  0.75  ]
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

```
[-0.5144  1.085   0.9089 -0.0841]
```

```python
print(t.add(20).numpy())
```

```
[19.4856 21.085  20.9089 19.9159]
```

```python
print(t.add(Tensor([[2.0], [3.5]])).numpy())
```

```
[[1.4856 3.085  2.9089 1.9159]
 [2.9856 4.585  4.4089 3.4159]]
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

```
[-0.5144  1.085   0.9089 -0.0841]
```

```python
print(t.sub(20).numpy())
```

```
[-20.5144 -18.915  -19.0911 -20.0841]
```

```python
print(t.sub(Tensor([[2.0], [3.5]])).numpy())
```

```
[[-2.5144 -0.915  -1.0911 -2.0841]
 [-4.0144 -2.415  -2.5911 -3.5841]]
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

```
[-0.5144  1.085   0.9089 -0.0841]
```

```python
print(t.mul(3).numpy())
```

```
[-1.5431  3.2549  2.7267 -0.2523]
```

```python
print(t.mul(Tensor([[-1.0], [2.0]])).numpy())
```

```
[[ 0.5144 -1.085  -0.9089  0.0841]
 [-1.0287  2.17    1.8178 -0.1682]]
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

```
[-0.5144  1.085   0.9089 -0.0841]
```

```python
print(t.div(3).numpy())
```

```
[-0.1715  0.3617  0.303  -0.028 ]
```

```python
print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
```

```
[0.5    1.3333 2.5   ]
```


#### idiv(x: Union[Self, float, int, bool], reverse: bool = False)

Divides `self` by `x`.
Equivalent to `self // x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.
`idiv` performs integer division (truncate towards zero).

```python
print(Tensor([-4, 7, 5, 4, -7, 8]).idiv(Tensor([2, -3, 8, -2, 3, 5])).numpy())
```

```
[-2 -2  0 -2 -2  1]
```


#### mod(x: 'Tensor | ConstType', reverse=False) -> Tensor

Mod `self` by `x`.
Equivalent to `self % x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.

```python
print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
```

```
[ 0 -2  5  0  2  3]
```


#### bitwise_xor(x: Union[Self, float, int, bool], reverse: bool = False)

Computes bitwise xor of `self` and `x`.
Equivalent to `self ^ x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([-1, -2, 3]).bitwise_xor(Tensor([1, 0, 3])).numpy())
```

```
[-2 -2  0]
```

```python
print(Tensor([True, True, False, False]).bitwise_xor(Tensor([True, False, True, False])).numpy())
```

```
[False  True  True False]
```


#### bitwise_and(x: Union[Self, float, int, bool], reverse: bool = False)

Computes the bitwise AND of `self` and `x`.
Equivalent to `self & x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
```

```
[ 2  4 16]
```

```python
print(Tensor([True, True, False, False]).bitwise_and(Tensor([True, False, True, False])).numpy())
```

```
[ True False False False]
```


#### bitwise_or(x: Union[Self, float, int, bool], reverse: bool = False)

Computes the bitwise OR of `self` and `x`.
Equivalent to `self | x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
```

```
[  6   5 255]
```

```python
print(Tensor([True, True, False, False]).bitwise_or(Tensor([True, False, True, False])).numpy())
```

```
[ True  True  True False]
```


#### bitwise_not() -> Tensor

Computes the bitwise NOT of `self`.
Equivalent to `~self`.

```python
print(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
```

```
[-1 -3 -6  0]
```

```python
print(Tensor([True, False]).bitwise_not().numpy())
```

```
[False  True]
```


#### lshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self << x`.

```python
print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
```

```
[  4  12 124]
```


#### rshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self >> x`.

```python
print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
```

```
[ 1  3 31]
```


#### pow(x: 'Tensor | ConstType', reverse=False) -> Tensor

Computes power of `self` with `x`.
Equivalent to `self ** x`.

```python
print(Tensor([-1, 2, 3]).pow(2.0).numpy())
```

```
[1 4 9]
```

```python
print(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
```

```
[-2147483648           1           5]
```

```python
print((2.0 ** Tensor([-1, 2, 3])).numpy())
```

```
[0.5 4.  8. ]
```


#### maximum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise maximum of `self` and `x`.

```python
print(Tensor([-1, 2, 3]).maximum(1).numpy())
```

```
[1 2 3]
```

```python
print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
```

```
[-1  2  9]
```


#### minimum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise minimum of `self` and `x`.

```python
print(Tensor([-1, 2, 3]).minimum(1).numpy())
```

```
[-1  1  1]
```

```python
print(Tensor([-1, 2, 3]).minimum(Tensor([-4, -2, 9])).numpy())
```

```
[-4 -2  3]
```


#### where(x: 'Tensor | ConstType | sint', y: 'Tensor | ConstType | sint') -> Tensor

Returns a tensor of elements selected from either `x` or `y`, depending on `self`.
`output_i = x_i if self_i else y_i`.

```python
cond = Tensor([[True, True, False], [True, False, False]])
print(cond.where(1, 3).numpy())
```

```
[[1 1 3]
 [1 3 3]]
```

```python
Tensor.manual_seed(42)
cond = Tensor.randn(2, 3)
print(cond.numpy())
```

```
[[ 0.9779  0.4678  0.5526]
 [-0.3288 -0.8555  0.2753]]
```

```python
print((cond > 0).where(cond, -float("inf")).numpy())
```

```
[[0.9779 0.4678 0.5526]
 [  -inf   -inf 0.2753]]
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

```
dtypes.float [-1.   2.5  3. ]
```

```python
t = t.cast(dtypes.int32)
print(t.dtype, t.numpy())
```

```
dtypes.int [-1  2  3]
```

```python
t = t.cast(dtypes.uint8)
print(t.dtype, t.numpy())
```

```
dtypes.uchar [255   2   3]
```


#### bitcast(dtype: 'DTypeLike') -> Tensor

Bitcasts `self` to the given `dtype` of the same itemsize.

`self` must not require a gradient.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```

```
dtypes.int [-1  2  3]
```

```python
t = t.bitcast(dtypes.uint32)
print(t.dtype, t.numpy())
```

```
dtypes.uint [4294967295          2          3]
```


#### float() -> Tensor

Convenience method to cast `self` to a `float32` Tensor.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```

```
dtypes.int [-1  2  3]
```

```python
t = t.float()
print(t.dtype, t.numpy())
```

```
dtypes.float [-1.  2.  3.]
```


#### half() -> Tensor

Convenience method to cast `self` to a `float16` Tensor.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```

```
dtypes.int [-1  2  3]
```

```python
t = t.half()
print(t.dtype, t.numpy())
```

```
dtypes.half [-1.  2.  3.]
```


#### int() -> Tensor

Convenience method to cast `self` to a `int32` Tensor.

```python
t = Tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
print(t.dtype, t.numpy())
```

```
dtypes.float [-1.5 -0.5  0.   0.5  1.5]
```

```python
t = t.int()
print(t.dtype, t.numpy())
```

```
dtypes.int [-1  0  0  0  1]
```


#### bool() -> Tensor

Convenience method to cast `self` to a `bool` Tensor.

```python
t = Tensor([-1, 0, 1])
print(t.dtype, t.numpy())
```

```
dtypes.int [-1  0  1]
```

```python
t = t.bool()
print(t.dtype, t.numpy())
```

```
dtypes.bool [ True False  True]
```


#### bfloat16() -> Tensor


#### double() -> Tensor


#### long() -> Tensor


#### short() -> Tensor

