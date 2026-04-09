## Movement (low level)

#### view(shape, *args) -> Self

`.view` is an alias for `.reshape`.


#### reshape(shape, *args) -> Self

Returns a tensor with the same data as the original tensor but with a different shape.
`shape` can be passed as a tuple or as separate arguments.


#### expand(shape, *args) -> Self

Returns a tensor that is expanded to the shape that is specified.
Expand can also increase the number of dimensions that a tensor has.

Passing a `-1` or `None` to a dimension means that its size will not be changed.


#### permute(order, *args) -> Self

Returns a tensor that is a permutation of the original tensor.
The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
`order` can be passed as a tuple or as separate arguments.


#### flip(axis, *args) -> Self

Returns a tensor that reverses the order of the original tensor along given `axis`.
`axis` can be passed as a tuple or as separate arguments.


#### shrink(arg: tuple[tuple['UOp | int', 'UOp | int'] | None, ...]) -> Self

Returns a tensor that shrinks the each axis based on input arg.
`arg` must have the same length as `self.ndim`.
For each axis, it can be `None`, which means no shrink, or a tuple `(start, end)` that works the same as Python slice.


#### pad(padding: 'Sequence[sint] | Sequence[tuple[sint, sint] | None]', mode: 'str' = 'constant', value: 'float' = 0.0) -> Tensor

Returns a tensor with padding applied based on the input `padding`.

`padding` supports two padding structures:

1. Flat padding: `(padding_left, padding_right, padding_top, padding_bottom, ...)`
    - This structure matches PyTorch's pad.
    - `padding` length must be even.

2. Group padding: `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
    - This structure matches pad for JAX, NumPy, TensorFlow, and others.
    - For each axis, padding can be `None`, meaning no padding, or a tuple `(start, end)`.
    - `padding` must have the same length as `self.ndim`.

Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
Padding modes is selected with `mode` which supports `constant`, `reflect` and `replicate`.


## Movement (high level)

#### __getitem__(indices) -> Tensor

Retrieves a sub-tensor using indexing.

Supported Index Types: `int | slice | Tensor | None | list | tuple | Ellipsis`

Examples:


#### gather(dim: 'int', index: 'Tensor') -> Tensor

Gathers values along an axis specified by `dim`.


#### cat(*args: 'Tensor', dim: 'int' = 0) -> Tensor

Concatenates self with other `Tensor` in `args` along an axis specified by `dim`.
All tensors must have the same shape except in the concatenating dimension.


#### stack(*args: 'Tensor', dim: 'int' = 0) -> Tensor

Concatenates self with other `Tensor` in `args` along a new dimension specified by `dim`.


#### repeat(repeats, *args) -> Self

Repeats tensor number of times along each dimension specified by `repeats`.
`repeats` can be passed as a tuple or as separate arguments.


#### repeat_interleave(repeats: int, dim: int | None = None) -> Self

Repeats elements of a tensor.


#### split(sizes: 'int | Sequence[int]', dim: 'int' = 0) -> tuple[Tensor, ...]

Splits the tensor into chunks along the dimension specified by `dim`.
If `sizes` is an integer, it splits into equally sized chunks if possible, otherwise the last chunk will be smaller.
If `sizes` is a list, it splits into `len(sizes)` chunks with size in `dim` according to `size`.


#### chunk(chunks: 'int', dim: 'int' = 0) -> list[Tensor]

Splits the tensor into `chunks` number of chunks along the dimension `dim`.
If the tensor size along `dim` is not divisible by `chunks`, all returned chunks will be the same size except the last one.
The function may return fewer than the specified number of chunks.


#### unfold(dim: 'int', size: 'sint', step: 'int') -> Tensor

Unfolds the tensor along dimension `dim` into overlapping windows.

Each window has length `size` and begins every `step` elements of `self`.
Returns the input tensor with dimension `dim` replaced by dims `(n_windows, size)`
where `n_windows = (self.shape[dim] - size) // step + 1`.


#### meshgrid(*args: 'Tensor', indexing: "Literal['ij', 'xy']" = 'ij') -> tuple[Tensor, ...]

Generates coordinate matrices from coordinate vectors.
Input tensors can be scalars or 1D tensors.

`indexing` determines how the output grids are aligned.
`ij` indexing follows matrix-style indexing and `xy` indexing follows Cartesian-style indexing.


#### squeeze(dim: int | None = None) -> Self

Returns a tensor with specified dimensions of input of size 1 removed.
If `dim` is not specified, all dimensions with size 1 are removed.


#### unsqueeze(dim: int) -> Self

Returns a tensor with a new dimension of size 1 inserted at the specified `dim`.


#### T -> typing.Self

`.T` is an alias for `.transpose()`.


#### transpose(dim0=1, dim1=0) -> Self

Returns a tensor that is a transposed version of the original tensor.
The given dimensions `dim0` and `dim1` are swapped.


#### flatten(start_dim=0, end_dim=-1) -> Self

Flattens the tensor by reshaping it into a one-dimensional tensor.
If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.


#### unflatten(dim: int, sizes: tuple[int, ...]) -> Self

Unflattens dimension `dim` of the tensor into multiple dimensions specified by `sizes`. `Tensor.flatten()` is the inverse of this function.


#### diag() -> Tensor

Returns a 2-D square tensor with the elements of input as the main diagonal.


#### diagonal() -> Tensor

Returns a view of input tensor with its main diagonal elements.


#### roll(shifts: 'int | tuple[int, ...]', dims: 'int | tuple[int, ...] | None' = None) -> Tensor

Rolls the tensor along specified dimension(s).
The rolling operation is circular, meaning that elements that go beyond the edge are wrapped around to the beginning of the dimension.


#### rearrange(formula: str, **sizes) -> Self

Rearranges input according to formula

See: https://einops.rocks/api/rearrange/

