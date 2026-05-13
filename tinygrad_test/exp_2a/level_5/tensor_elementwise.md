Elementwise ops operate on a per element basis. They don't change the shape of the tensor.

## Unary Ops (math)

#### Tensor.logical_not() -> Tensor

Computes the logical NOT of the tensor element-wise.

```python
print(Tensor([False, True]).logical_not().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.neg() -> Tensor

Negates the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.log() -> Tensor

Computes the natural logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm

```python
print(Tensor([1., 2., 4., 8.]).log().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.log2() -> Tensor

Computes the base-2 logarithm element-wise.

See: https://en.wikipedia.org/wiki/Logarithm

```python
print(Tensor([1., 2., 4., 8.]).log2().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### log10 *(resolution failed)*

#### Tensor.exp() -> Tensor

Computes the exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function

```python
print(Tensor([0., 1., 2., 3.]).exp().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.exp2() -> Tensor

Computes the base-2 exponential function element-wise.

See: https://en.wikipedia.org/wiki/Exponential_function

```python
print(Tensor([0., 1., 2., 3.]).exp2().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.sqrt() -> Tensor

Computes the square root of the tensor element-wise.

```python
print(Tensor([1., 2., 3., 4.]).sqrt().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.rsqrt()

Computes the reciprocal of the square root of the tensor element-wise.

```python
print(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.sin() -> Tensor

Computes the sine of the tensor element-wise.

```python
print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
```

```
[execution error: name 'math' is not defined]
```


#### Tensor.cos() -> Tensor

Computes the cosine of the tensor element-wise.

```python
print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
```

```
[execution error: name 'math' is not defined]
```


#### Tensor.tan() -> Tensor

Computes the tangent of the tensor element-wise.

```python
print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
```

```
[execution error: name 'math' is not defined]
```


#### Tensor.asin() -> Tensor

Computes the inverse sine (arcsine) of the tensor element-wise.

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).asin().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.acos() -> Tensor

Computes the inverse cosine (arccosine) of the tensor element-wise.

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).acos().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.atan() -> Tensor

Computes the inverse tangent (arctan) of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).atan().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.trunc()

Truncates the tensor element-wise.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).trunc().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.ceil()

Rounds the tensor element-wise towards positive infinity.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).ceil().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.floor()

Rounds the tensor element-wise towards negative infinity.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).floor().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.round() -> Tensor

Rounds the tensor element-wise with rounding half to even.

```python
print(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]).round().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.isinf(detect_positive: bool = True, detect_negative: bool = True)

Checks the tensor element-wise to return True where the element is infinity, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.isnan()

Checks the tensor element-wise to return True where the element is NaN, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.isfinite()

Checks the tensor element-wise to return True where the element is finite, otherwise returns False

```python
print(Tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.lerp(end: 'Tensor', weight: 'Tensor | float') -> Tensor

Linearly interpolates between `self` and `end` by `weight`.

```python
print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.square()

Squares the tensor element-wise.
Equivalent to `self*self`.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.clamp(min_=None, max_=None)

Clips (clamps) the values in the tensor between `min_` and `max_` element-wise.
If `min_` is `None`, there is no lower bound. If `max_` is None, there is no upper bound.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.clip(min_=None, max_=None)

Alias for `Tensor.clamp`.


#### Tensor.sign() -> Tensor

Returns the sign of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.abs() -> Tensor

Computes the absolute value of the tensor element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.reciprocal() -> Tensor

Computes `1/x` element-wise.

```python
print(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


## Unary Ops (activation)

#### Tensor.relu()

Applies the Rectified Linear Unit (ReLU) function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.sigmoid()

Applies the Sigmoid function element-wise.

- Described: https://en.wikipedia.org/wiki/Sigmoid_function

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.logsigmoid() -> Tensor

Applies the LogSigmoid function element-wise.

- See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).logsigmoid().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.hardsigmoid(alpha: float = 0.16666666666666666, beta: float = 0.5)

Applies the Hardsigmoid function element-wise.
NOTE: default `alpha` and `beta` values are taken from torch

- See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardsigmoid().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.elu(alpha=1.0) -> Tensor

Applies the Exponential Linear Unit (ELU) function element-wise.

- Paper: https://arxiv.org/abs/1511.07289v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.celu(alpha=1.0) -> Tensor

Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

- Paper: https://arxiv.org/abs/1704.07483

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.selu(alpha=1.67326, gamma=1.0507) -> Tensor

Applies the Scaled Exponential Linear Unit (SELU) function element-wise.

- Paper: https://arxiv.org/abs/1706.02515v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).selu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.swish()

See `.silu()`

- Paper: https://arxiv.org/abs/1710.05941v1

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.silu()

Applies the Sigmoid Linear Unit (SiLU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.relu6()

Applies the ReLU6 function element-wise.

- Paper: https://arxiv.org/abs/1704.04861v1

```python
print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.hardswish()

Applies the Hardswish function element-wise.

- Paper: https://arxiv.org/abs/1905.02244v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.tanh()

Applies the Hyperbolic Tangent (tanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.sinh() -> Tensor

Applies the Hyperbolic Sine (sinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.cosh() -> Tensor

Applies the Hyperbolic Cosine (cosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.atanh() -> Tensor

Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

```python
print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.asinh() -> Tensor

Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.acosh() -> Tensor

Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

- Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.hardtanh(min_val=-1, max_val=1)

Applies the Hardtanh function element-wise.

```python
print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.erf() -> Tensor

Applies error function element-wise.

- Described: https://en.wikipedia.org/wiki/Error_function

```python
print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).erf().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.gelu()

Applies the Gaussian Error Linear Unit (GELU) function element-wise.

- Paper: https://arxiv.org/abs/1606.08415v5

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.quick_gelu()

Applies the Sigmoid GELU approximation element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.leaky_relu(neg_slope=0.01)

Applies the Leaky ReLU function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leaky_relu(neg_slope=0.42).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.mish() -> Tensor

Applies the Mish function element-wise.

- Paper: https://arxiv.org/abs/1908.08681v3

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.softplus(beta=1.0) -> Tensor

Applies the Softplus function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.softsign() -> Tensor

Applies the Softsign function element-wise.

```python
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


## Elementwise Ops (broadcasted)

#### Tensor.add(x: Self | float | int | bool, reverse: bool = False)

Adds `self` and `x`.
Equivalent to `self + x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.add(20).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.add(Tensor([[2.0], [3.5]])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.sub(x: 'Tensor | ConstType', reverse=False) -> Tensor

Subtracts `x` from `self`.
Equivalent to `self - x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.sub(20).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.sub(Tensor([[2.0], [3.5]])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.mul(x: Self | float | int | bool, reverse: bool = False)

Multiplies `self` and `x`.
Equivalent to `self * x`.
Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

```python
Tensor.manual_seed(42)
t = Tensor.randn(4)
print(t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.mul(3).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.mul(Tensor([[-1.0], [2.0]])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.div(x: 'Tensor | ConstType', reverse=False, rounding_mode: "Literal['trunc', 'floor'] | None" = None) -> Tensor

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
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(t.div(3).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.idiv(x: Self | float | int | bool, reverse: bool = False)

Divides `self` by `x`.
Equivalent to `self // x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.
`idiv` performs integer division (truncate towards zero).

```python
print(Tensor([-4, 7, 5, 4, -7, 8]).idiv(Tensor([2, -3, 8, -2, 3, 5])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.mod(x: 'Tensor | ConstType', reverse=False) -> Tensor

Mod `self` by `x`.
Equivalent to `self % x`.
Supports broadcasting to a common shape, type promotion, and integer inputs.

```python
print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bitwise_xor(x: Self | float | int | bool, reverse: bool = False)

Computes bitwise xor of `self` and `x`.
Equivalent to `self ^ x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([-1, -2, 3]).bitwise_xor(Tensor([1, 0, 3])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([True, True, False, False]).bitwise_xor(Tensor([True, False, True, False])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bitwise_and(x: Self | float | int | bool, reverse: bool = False)

Computes the bitwise AND of `self` and `x`.
Equivalent to `self & x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([True, True, False, False]).bitwise_and(Tensor([True, False, True, False])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bitwise_or(x: Self | float | int | bool, reverse: bool = False)

Computes the bitwise OR of `self` and `x`.
Equivalent to `self | x`.
Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

```python
print(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([True, True, False, False]).bitwise_or(Tensor([True, False, True, False])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bitwise_not() -> Tensor

Computes the bitwise NOT of `self`.
Equivalent to `~self`.

```python
print(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([True, False]).bitwise_not().numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.lshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self << x`.

```python
print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.rshift(x: 'Tensor | int', reverse=False) -> Tensor

Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
Equivalent to `self >> x`.

```python
print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.pow(x: 'Tensor | ConstType', reverse=False) -> Tensor

Computes power of `self` with `x`.
Equivalent to `self ** x`.

```python
print(Tensor([-1, 2, 3]).pow(2.0).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print((2.0 ** Tensor([-1, 2, 3])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.maximum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise maximum of `self` and `x`.

```python
print(Tensor([-1, 2, 3]).maximum(1).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.minimum(x: 'Tensor | ConstType') -> Tensor

Computes element-wise minimum of `self` and `x`.

```python
print(Tensor([-1, 2, 3]).minimum(1).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print(Tensor([-1, 2, 3]).minimum(Tensor([-4, -2, 9])).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.where(x: 'Tensor | ConstType | sint', y: 'Tensor | ConstType | sint') -> Tensor

Returns a tensor of elements selected from either `x` or `y`, depending on `self`.
`output_i = x_i if self_i else y_i`.

```python
cond = Tensor([[True, True, False], [True, False, False]])
print(cond.where(1, 3).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
Tensor.manual_seed(42)
cond = Tensor.randn(2, 3)
print(cond.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
print((cond > 0).where(cond, -float("inf")).numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.copysign(other) -> Tensor

Returns a tensor of with the magnitude of `self` and the sign of `other`, elementwise.


#### Tensor.logaddexp(other) -> Tensor

Calculates (self.exp()+other.exp()).log(), elementwise.


## Casting Ops

#### Tensor.cast(dtype: 'DTypeLike') -> Tensor

Casts `self` to the given `dtype`.

```python
t = Tensor([-1, 2.5, 3], dtype=dtypes.float)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.cast(dtypes.int32)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.cast(dtypes.uint8)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bitcast(dtype: 'DTypeLike') -> Tensor

Bitcasts `self` to the given `dtype` of the same itemsize.

`self` must not require a gradient.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.bitcast(dtypes.uint32)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.float() -> Tensor

Convenience method to cast `self` to a `float32` Tensor.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.float()
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.half() -> Tensor

Convenience method to cast `self` to a `float16` Tensor.

```python
t = Tensor([-1, 2, 3], dtype=dtypes.int32)
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.half()
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.int() -> Tensor

Convenience method to cast `self` to a `int32` Tensor.

```python
t = Tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.int()
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bool() -> Tensor

Convenience method to cast `self` to a `bool` Tensor.

```python
t = Tensor([-1, 0, 1])
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```

```python
t = t.bool()
print(t.dtype, t.numpy())
```

```
[execution error: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
The following compiled module files exist, but seem incompatible
with with either python 'cpython-314' or the platform 'linux':

  * _multiarray_umath.cpython-313-x86_64-linux-gnu.so

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python 3.14 from "/usr/bin/python3"
  * The NumPy version is: "2.4.4"

and make sure that they are the versions you expect.

Please carefully study the information and documentation linked above.
This is unlikely to be a NumPy issue but will be caused by a bad install
or environment on your machine.

Original error was: No module named 'numpy._core._multiarray_umath'
]
```


#### Tensor.bfloat16() -> Tensor


#### Tensor.double() -> Tensor


#### Tensor.long() -> Tensor


#### Tensor.short() -> Tensor

