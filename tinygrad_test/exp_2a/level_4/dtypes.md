#### DType(priority: 'int', bitsize: 'int', name: 'str', fmt: 'FmtStr | None', count: 'int', _scalar: 'DType | None')

DType(*args, **kwargs)


#### Common dtypes

| dtype | description |
|-------|-------------|
| `dtypes.float32` | 32-bit floating point (default) |
| `dtypes.float16` | 16-bit floating point |
| `dtypes.bfloat16` | 16-bit brain float |
| `dtypes.float64` | 64-bit floating point |
| `dtypes.int8` | 8-bit signed integer |
| `dtypes.int16` | 16-bit signed integer |
| `dtypes.int32` | 32-bit signed integer |
| `dtypes.int64` | 64-bit signed integer |
| `dtypes.uint8` | 8-bit unsigned integer |
| `dtypes.uint16` | 16-bit unsigned integer |
| `dtypes.uint32` | 32-bit unsigned integer |
| `dtypes.uint64` | 64-bit unsigned integer |
| `dtypes.bool` | boolean |


#### ConstType(*args, **kwargs)

Represent a PEP 604 union type

E.g. for int | str

