# Tinygrad Special Instructions: Lazy Execution & Advanced Usage

## Lazy Execution Model

Tinygrad uses a **lazy execution model**. Operations on `Tensor` objects do not
immediately compute results — they build a computation graph that is only
evaluated (realized) when explicitly triggered.

### When computation actually happens

Computation is triggered by:
- **`.numpy()`** — realizes the tensor and returns a NumPy array
- **`.realize()`** — forces evaluation and returns the `Tensor` itself (stays on device)
- **`.item()`** — realizes a scalar tensor and returns a Python scalar
- **`.tolist()`** — realizes and returns a nested Python list

```python
from tinygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])
y = x * 2          # nothing computed yet — lazy
z = y + 1          # still lazy
result = z.numpy()  # NOW computation happens
```

### Common gotchas

1. **Printing a tensor does not realize it.**
   ```python
   print(x)        # shows <Tensor ...> — does NOT show values
   print(x.numpy()) # shows values
   ```

2. **Modifying a NumPy array after `.numpy()` has no effect on the tensor.**
   The returned array is a copy, not a view.

3. **Don't call `.numpy()` inside a hot loop if you can avoid it.**
   Each `.numpy()` call flushes the lazy graph. Accumulate operations and
   realize once at the end.

4. **Shape errors are deferred.** An invalid reshape may not raise until
   `.realize()` or `.numpy()` is called.

### `.realize()` vs `.numpy()`

| | `.realize()` | `.numpy()` |
|---|---|---|
| Returns | `Tensor` | `np.ndarray` |
| Data stays on device | Yes | No (copies to CPU) |
| Use when | chaining further ops | reading values |

```python
t = Tensor.rand(4, 4).realize()  # force eval, keep on GPU
arr = t.numpy()                  # copy to CPU
```

---

## `Tensor.training` Context Manager

tinygrad does **not** use `model.train()` / `model.eval()` calls.
Instead, use the `Tensor.train()` context manager.

```python
with Tensor.train():
    out = model(x)          # dropout, batchnorm use training behavior
    loss = out.mean()
    loss.backward()
    opt.step()

# Outside the context, Tensor.training == False (inference mode)
out = model(x)              # dropout disabled, batchnorm uses running stats
```

- `Tensor.training` is a class-level boolean flag.
- `with Tensor.train():` sets it to `True` on entry and restores it on exit.
- Layers like `BatchNorm` and `Dropout` check this flag.

---

## JIT Usage Notes

The `TinyJit` decorator compiles a function's kernel sequence and replays it
for subsequent calls with the same shapes.

```python
from tinygrad import TinyJit, Tensor

@TinyJit
def forward(x: Tensor) -> Tensor:
    return model(x).realize()  # .realize() is required for JIT output
```

### JIT constraints

- **Input tensors must be realized** before passing to a JIT function.
- **Output must be realized** inside the JIT function (call `.realize()`).
- **Fixed shapes only** — JIT breaks if input shapes change between calls.
- The first two calls are "warmup" — the graph is captured on call 3+.
- Non-tinygrad operations (Python control flow depending on tensor values,
  random seeds that change each call, etc.) may not work correctly.

### Resetting the JIT

```python
forward.reset()  # clears the captured graph, allowing re-capture
```

---

## Additional Tips

- **`Tensor.no_grad()`** — context manager to disable gradient tracking
  (analogous to `torch.no_grad()`).
  ```python
  with Tensor.no_grad():
      out = model(x)
  ```

- **Device placement** — specify device at creation time:
  ```python
  x = Tensor([1.0, 2.0], device="GPU")
  x = x.to("CPU")
  ```

- **In-place device transfer** — use `.to_()` to move a tensor in-place
  (mutates the tensor's device attribute).

- **`.detach()`** — returns a new tensor with no gradient history.
  Use before passing tensors to non-differentiable operations.

- **`.contiguous()`** — ensures the tensor's underlying buffer is
  contiguous in memory. Call before passing to C extensions or when
  debugging layout issues.
