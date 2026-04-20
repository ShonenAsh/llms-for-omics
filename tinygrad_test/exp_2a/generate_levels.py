"""
generate_levels.py
==================
Generates 5 documentation-level folders for a tinygrad LLM benchmarking ablation study.

Each level provides progressively more context:
  Level 1 — API signatures only
  Level 2 — signatures + description
  Level 3 — signatures + description + first code example; quickstart.md included
  Level 4 — signatures + description + all examples with executed output; quickstart.md included
  Level 5 — Level 4 + special_instructions.md + pytorch_migration.md
"""

import sys
import os
import re
import io
import inspect
import contextlib
import pathlib
import textwrap

# ── Paths ──────────────────────────────────────────────────────────────────────
VENV_SITE = "/home/shonenash/data/workspace/neu-ra/llms-for-omics/tinygrad_test/.venv/lib/python3.13/site-packages"
sys.path.insert(0, VENV_SITE)

DOCS_DIR = pathlib.Path("/home/shonenash/data/workspace/neu-ra/llms-for-omics/tinygrad_test/exp_2a/docs")
OUT_BASE = pathlib.Path("/home/shonenash/data/workspace/neu-ra/llms-for-omics/tinygrad_test/exp_2a")

# Files that have ::: directives and should be processed
DIRECTIVE_FILES = [
    "tensor/creation.md",
    "tensor/elementwise.md",
    "tensor/index.md",
    "tensor/movement.md",
    "tensor/ops.md",
    "tensor/properties.md",
    "nn.md",
    "dtypes.md",
]
# Static prose file included as-is in levels 3-5
QUICKSTART_FILE = "quickstart.md"

# ── Import tinygrad ────────────────────────────────────────────────────────────
from tinygrad import Tensor, dtypes
import tinygrad.nn as nn
import tinygrad.nn.optim as optim
import tinygrad.nn.state as state
from tinygrad.dtype import DType, ConstType

# ── Code-block regex ───────────────────────────────────────────────────────────
# Matches ```python <attrs>\n<code>\n```
CODE_BLOCK_RE = re.compile(
    r'```python(?P<attrs>[^\n]*)\n(?P<code>.*?)```',
    re.DOTALL
)


def parse_attrs(attr_str: str) -> dict:
    """Parse mkdocs-style attribute string into a dict of key=value pairs."""
    attrs = {}
    for m in re.finditer(r'(\w+)="([^"]*)"', attr_str):
        attrs[m.group(1)] = m.group(2)
    return attrs


# ── Object resolution ──────────────────────────────────────────────────────────

CURATED_DTYPES = """\
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
"""


def resolve_object(dotted: str):
    """Resolve a dotted name like 'tinygrad.Tensor.relu' to a Python object.
    Returns (obj, kind) where kind is 'method', 'classmethod', 'property',
    'class', or 'function'.
    """
    # Strip leading 'tinygrad.'
    parts = dotted.split(".")
    # try to navigate from known roots
    # We support: tinygrad.Tensor.*, tinygrad.nn.*, tinygrad.nn.optim.*,
    #             tinygrad.nn.state.*, tinygrad.dtype.*

    if parts[0] == "tinygrad":
        parts = parts[1:]

    # Build lookup
    namespace = {
        "Tensor": Tensor,
        "nn": nn,
        "dtype": sys.modules.get("tinygrad.dtype"),
    }

    obj = None
    try:
        if parts[0] == "Tensor":
            if len(parts) == 1:
                return Tensor, "class"
            attr = parts[1]
            raw = Tensor.__dict__.get(attr)
            if raw is None:
                # Try getattr (picks up inherited / descriptors)
                raw = getattr(Tensor, attr, None)
            if isinstance(raw, property):
                return raw, "property"
            obj = raw
            if obj is None:
                return None, "unknown"
            if isinstance(obj, classmethod):
                return obj.__func__, "classmethod"
            return obj, "method"

        elif parts[0] == "nn":
            if len(parts) == 1:
                return nn, "module"
            sub = parts[1]
            if sub == "optim":
                if len(parts) == 3:
                    obj = getattr(optim, parts[2], None)
                    return obj, "function"
                return optim, "module"
            elif sub == "state":
                if len(parts) == 3:
                    obj = getattr(state, parts[2], None)
                    return obj, "function"
                return state, "module"
            else:
                obj = getattr(nn, sub, None)
                if obj is None:
                    return None, "unknown"
                if inspect.isclass(obj):
                    return obj, "class"
                return obj, "function"

        elif parts[0] == "dtype":
            if len(parts) == 2:
                name = parts[1]
                if name == "dtypes":
                    return "SPECIAL_DTYPES", "special_dtypes"
                mod = sys.modules.get("tinygrad.dtype")
                obj = getattr(mod, name, None)
                return obj, "class"
    except Exception:
        pass

    return None, "unknown"


def get_signature_str(dotted: str, obj, kind: str) -> str:
    """Return a formatted signature string for the object."""
    name = dotted.split(".")[-1]

    if kind == "special_dtypes":
        return None  # handled separately

    if kind == "class":
        # Use __init__ signature, stripping self
        try:
            init = obj.__init__
            sig = inspect.signature(init)
            params = list(sig.parameters.values())
            # Remove 'self'
            params = [p for p in params if p.name != "self"]
            new_sig = sig.replace(parameters=params)
            # Clean up ContextVar defaults
            param_strs = []
            for p in params:
                ps = str(p)
                # Replace ContextVar objects with their name
                ps = re.sub(r'=<tinygrad\.helpers\.ContextVar object at 0x[0-9a-f]+>', '=<ContextVar>', ps)
                param_strs.append(ps)
            return f"#### {name}({', '.join(param_strs)})"
        except (ValueError, TypeError):
            return f"#### {name}(...)"

    if kind in ("method", "classmethod", "function"):
        try:
            sig = inspect.signature(obj)
            params = list(sig.parameters.values())
            # Remove 'self' for regular methods
            if params and params[0].name == "self":
                params = params[1:]
            # Clean up return annotation and ContextVar
            ret = ""
            if sig.return_annotation is not inspect.Parameter.empty:
                ret_str = str(sig.return_annotation)
                # Simplify 'typing.X' to 'X'
                ret_str = ret_str.replace("typing.", "")
                ret = f" -> {ret_str}"
            param_strs = []
            for p in params:
                ps = str(p)
                ps = re.sub(r'=<tinygrad\.helpers\.ContextVar object at 0x[0-9a-f]+>', '=<ContextVar>', ps)
                param_strs.append(ps)
            return f"#### {name}({', '.join(param_strs)}){ret}"
        except (ValueError, TypeError):
            return f"#### {name}(...)"

    if kind == "property":
        try:
            # property: show as attribute
            fget = obj.fget
            sig = inspect.signature(fget)
            ret = ""
            if sig.return_annotation is not inspect.Parameter.empty:
                ret = f" -> {sig.return_annotation}"
            return f"#### {name}{ret}"
        except Exception:
            return f"#### {name}"

    return f"#### {name}"


def get_description(obj, kind: str) -> str:
    """Extract description (everything before first code block) from docstring."""
    if kind == "property":
        doc = obj.fget.__doc__ if obj.fget else ""
    elif obj is None:
        return ""
    else:
        doc = getattr(obj, "__doc__", "") or ""

    if not doc:
        return ""
    doc = doc.strip()
    # Find first code block
    m = CODE_BLOCK_RE.search(doc)
    if m:
        desc = doc[:m.start()].strip()
    else:
        desc = doc.strip()
    return desc


def get_code_blocks(obj, kind: str) -> list:
    """Extract all code blocks from docstring as list of (attrs_dict, code) tuples."""
    if kind == "property":
        doc = obj.fget.__doc__ if obj.fget else ""
    elif obj is None:
        return []
    else:
        doc = getattr(obj, "__doc__", "") or ""

    if not doc:
        return []
    blocks = []
    for m in CODE_BLOCK_RE.finditer(doc):
        attrs = parse_attrs(m.group("attrs"))
        code = m.group("code")
        blocks.append((attrs, code))
    return blocks


def execute_code(code: str, session_ns: dict) -> str:
    """Execute code in session_ns, capture and return stdout."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, session_ns)  # noqa: S102
    except Exception as e:
        buf.write(f"[execution error: {e}]")
    return buf.getvalue()


# ── Per-directive rendering ────────────────────────────────────────────────────

def render_directive(dotted: str, options: dict, level: int, session_ns: dict) -> str:
    """Render a single ::: directive at the given documentation level."""
    obj, kind = resolve_object(dotted)

    # Special case: tinygrad.dtype.dtypes
    if kind == "special_dtypes":
        if level == 1:
            return "#### dtypes\n"
        else:
            return CURATED_DTYPES + "\n"

    if obj is None:
        return f"#### {dotted.split('.')[-1]} *(resolution failed)*\n"

    # Special case: tinygrad.Tensor (the class itself)
    if dotted == "tinygrad.Tensor" and kind == "class":
        desc = (Tensor.__doc__ or "").strip()
        # Remove setup code blocks (exec=true but no source=above) from description display
        # For the Tensor class, the __doc__ contains a setup block only
        blocks = get_code_blocks(Tensor, "class")
        # Find first non-setup block
        visible_blocks = [(a, c) for a, c in blocks if a.get("source") == "above"]
        setup_blocks = [(a, c) for a, c in blocks if a.get("source") != "above" and a.get("exec") == "true"]

        # Execute all setup blocks (all levels that do execution — 4+)
        if level >= 4:
            for attrs, code in setup_blocks:
                execute_code(code, session_ns)

        # Extract desc text (before first block)
        m = CODE_BLOCK_RE.search(desc)
        if m:
            text_desc = desc[:m.start()].strip()
        else:
            text_desc = desc.strip()

        parts = [f"## Tensor\n\n{text_desc}\n"]
        return "\n".join(parts) + "\n"

    # --- Normal directive ---
    sig_str = get_signature_str(dotted, obj, kind)
    if sig_str is None:
        return ""

    if level == 1:
        return sig_str + "\n\n"

    desc = get_description(obj, kind)
    blocks = get_code_blocks(obj, kind)

    # Separate visible vs setup-only
    visible_blocks = [(a, c) for a, c in blocks if a.get("source") == "above"]
    setup_blocks = [(a, c) for a, c in blocks if a.get("source") != "above" and a.get("exec") == "true"]

    # For level 4+: execute setup blocks to maintain session state
    if level >= 4:
        for attrs, code in setup_blocks:
            execute_code(code, session_ns)

    lines = [sig_str, ""]
    if desc:
        lines.append(desc)
        lines.append("")

    if level == 2:
        return "\n".join(lines) + "\n"

    # Level 3: first visible block only, no output
    if level == 3:
        if visible_blocks:
            attrs, code = visible_blocks[0]
            lines.append("```python")
            lines.append(code.rstrip())
            lines.append("```")
            lines.append("")
        return "\n".join(lines) + "\n"

    # Level 4+: all visible blocks with output
    if level >= 4:
        for attrs, code in visible_blocks:
            lines.append("```python")
            lines.append(code.rstrip())
            lines.append("```")
            # Execute in session and capture output if result requested
            output = execute_code(code, session_ns)
            if attrs.get("result") == "python" and output:
                lines.append("")
                lines.append("```")
                lines.append(output.rstrip())
                lines.append("```")
            lines.append("")

    return "\n".join(lines) + "\n"


# ── File processing ────────────────────────────────────────────────────────────

def process_file(rel_path: str, level: int) -> str:
    """Process a single documentation file, returning rendered markdown."""
    src = DOCS_DIR / rel_path
    text = src.read_text()
    lines = text.splitlines()

    output_parts = []
    session_ns: dict = {}

    # Pre-seed session with common imports (levels 3+)
    if level >= 3:
        setup_code = (
            "from tinygrad import Tensor, dtypes, nn\n"
            "import numpy as np\n"
            "import math\n"
            "np.set_printoptions(precision=4)\n"
        )
        execute_code(setup_code, session_ns)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for ::: directive
        if stripped.startswith(":::"):
            directive = stripped[3:].strip()
            # Collect options (indented lines following)
            options = {}
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Options are indented (4 spaces or a tab) under the directive
                if next_line and (next_line.startswith("    ") or next_line.startswith("\t")):
                    opt_stripped = next_line.strip()
                    if ":" in opt_stripped:
                        k, _, v = opt_stripped.partition(":")
                        options[k.strip()] = v.strip()
                    i += 1
                elif next_line.strip() == "":
                    # blank line might be end of options
                    i += 1
                    break
                else:
                    break
            rendered = render_directive(directive, options, level, session_ns)
            output_parts.append(rendered)
        else:
            output_parts.append(line)
            i += 1

    return "\n".join(output_parts)


# ── Special files for Level 5 ──────────────────────────────────────────────────

SPECIAL_INSTRUCTIONS_MD = """\
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
"""

PYTORCH_MIGRATION_MD = """\
# PyTorch → Tinygrad Migration Guide

This guide covers common patterns and their tinygrad equivalents.

---

## Tensor Creation

| PyTorch | Tinygrad |
|---------|----------|
| `torch.tensor([1,2,3])` | `Tensor([1,2,3])` |
| `torch.zeros(2, 3)` | `Tensor.zeros(2, 3)` |
| `torch.ones(2, 3)` | `Tensor.ones(2, 3)` |
| `torch.rand(2, 3)` | `Tensor.rand(2, 3)` |
| `torch.randn(2, 3)` | `Tensor.randn(2, 3)` |
| `torch.arange(0, 10)` | `Tensor.arange(0, 10)` |
| `torch.eye(3)` | `Tensor.eye(3)` |
| `torch.full((2,3), 5.0)` | `Tensor.full((2,3), 5.0)` |
| `torch.from_numpy(arr)` | `Tensor(arr)` |

```python
import numpy as np
from tinygrad import Tensor

arr = np.array([1.0, 2.0, 3.0])
t = Tensor(arr)          # wraps numpy array
t2 = Tensor([1, 2, 3])  # from Python list
```

---

## Device Placement

```python
# PyTorch
x = torch.tensor([1.0]).to("cuda")
x = torch.tensor([1.0]).cuda()

# Tinygrad — specify at creation or use .to()
x = Tensor([1.0], device="GPU")
x = Tensor([1.0]).to("GPU")
x = x.to("CPU")   # move to CPU
x.to_("GPU")       # in-place move
```

Tinygrad device strings: `"CPU"`, `"GPU"`, `"METAL"`, `"CUDA"`, `"CLANG"`.

---

## Training Mode

```python
# PyTorch
model.train()
# ... training loop ...
model.eval()

# Tinygrad — use context manager
from tinygrad import Tensor

with Tensor.train():
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    opt.step()

# Inference (outside context) — Tensor.training == False automatically
out = model(x)
```

---

## Gradient Management

```python
# PyTorch
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Tinygrad
opt.zero_grad()
loss.backward()
opt.step()
```

```python
# PyTorch no_grad context
with torch.no_grad():
    out = model(x)

# Tinygrad
with Tensor.no_grad():
    out = model(x)
```

---

## Optimizers

```python
from tinygrad.nn.optim import SGD, Adam, AdamW

params = nn.state.get_parameters(model)

# SGD
opt = SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam
opt = Adam(params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)

# AdamW
opt = AdamW(params, lr=1e-3, weight_decay=0.01)
```

| PyTorch | Tinygrad |
|---------|----------|
| `torch.optim.SGD(params, lr, momentum, weight_decay)` | `SGD(params, lr, momentum, weight_decay)` |
| `torch.optim.Adam(params, lr, betas, eps)` | `Adam(params, lr, b1, b2, eps)` |
| `torch.optim.AdamW(params, lr, weight_decay)` | `AdamW(params, lr, weight_decay)` |

---

## Neural Network Layers

### Linear

```python
# PyTorch
import torch.nn as torch_nn
lin = torch_nn.Linear(in_features=128, out_features=64)

# Tinygrad
from tinygrad import nn
lin = nn.Linear(128, 64)

# Usage
x = Tensor.rand(8, 128)
out = lin(x)   # shape: (8, 64)
```

### BatchNorm

```python
# PyTorch
bn = torch_nn.BatchNorm2d(num_features=32)

# Tinygrad — BatchNorm handles 1D, 2D, 3D inputs based on shape
bn = nn.BatchNorm(32)

# Usage — expects (N, C) or (N, C, *spatial)
x = Tensor.rand(8, 32, 4, 4)
out = bn(x)
```

### LayerNorm

```python
# PyTorch
ln = torch_nn.LayerNorm(normalized_shape=64)

# Tinygrad
ln = nn.LayerNorm(64)

x = Tensor.rand(8, 10, 64)
out = ln(x)
```

### Embedding

```python
# PyTorch
emb = torch_nn.Embedding(num_embeddings=1000, embedding_dim=128)

# Tinygrad
emb = nn.Embedding(1000, 128)

idx = Tensor([[1, 2, 3], [4, 5, 6]])  # integer tensor
out = emb(idx)   # shape: (2, 3, 128)
```

### Conv2d

```python
# PyTorch
conv = torch_nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Tinygrad
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

x = Tensor.rand(1, 3, 32, 32)
out = conv(x)   # shape: (1, 16, 32, 32)
```

---

## Loss Functions

```python
# PyTorch                              # Tinygrad
# MSE
loss = torch_nn.functional.mse_loss(out, target)
loss = ((out - target) ** 2).mean()

# Binary cross-entropy
loss = torch_nn.functional.binary_cross_entropy(out, target)
loss = out.binary_crossentropy(target)

# Binary cross-entropy with logits
loss = torch_nn.functional.binary_cross_entropy_with_logits(logits, target)
loss = logits.binary_crossentropy_logits(target)

# Cross-entropy (sparse labels)
loss = torch_nn.functional.cross_entropy(logits, labels)
loss = logits.sparse_categorical_crossentropy(labels)
```

---

## Model Save / Load

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))

# Tinygrad — uses safetensors format
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

state_dict = get_state_dict(model)
safe_save(state_dict, "model.safetensors")

# Load back
state_dict = safe_load("model.safetensors")
load_state_dict(model, state_dict)
```

---

## Getting Model Parameters

```python
# PyTorch
params = list(model.parameters())

# Tinygrad
from tinygrad.nn.state import get_parameters
params = get_parameters(model)
```

---

## Tensor Operations Cheat Sheet

| PyTorch | Tinygrad |
|---------|----------|
| `x.reshape(2, -1)` | `x.reshape(2, -1)` |
| `x.view(2, -1)` | `x.reshape(2, -1)` or `x.view(2, -1)` |
| `x.transpose(0, 1)` | `x.transpose(0, 1)` |
| `x.permute(2, 0, 1)` | `x.permute(2, 0, 1)` |
| `torch.cat([a, b], dim=0)` | `a.cat(b, dim=0)` or `Tensor.cat(a, b, dim=0)` |
| `torch.stack([a, b])` | `Tensor.stack([a, b])` |
| `x.squeeze(0)` | `x.squeeze(0)` |
| `x.unsqueeze(0)` | `x.unsqueeze(0)` |
| `x.mean(dim=-1)` | `x.mean(axis=-1)` |
| `x.sum(dim=0)` | `x.sum(axis=0)` |
| `x.max(dim=1)` | `x.max(axis=1)` |
| `x.softmax(dim=-1)` | `x.softmax(axis=-1)` |
| `x.detach()` | `x.detach()` |
| `x.numpy()` | `x.numpy()` |
| `x.item()` | `x.item()` |
| `x.dtype` | `x.dtype` |
| `x.shape` | `x.shape` |
| `x.device` | `x.device` |
| `x.float()` | `x.float()` |
| `x.half()` | `x.half()` |

---

## Full Training Loop Example

```python
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save


class SimpleMLP:
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.fc1(x).relu()
        return self.fc2(x)


model = SimpleMLP()
opt = Adam(get_parameters(model), lr=1e-3)

# Training
with Tensor.train():
    for step in range(100):
        x = Tensor.rand(32, 784)
        y = Tensor.randint(32, low=0, high=10)

        logits = model(x)
        loss = logits.sparse_categorical_crossentropy(y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step}, loss={loss.numpy():.4f}")

# Inference
out = model(Tensor.rand(8, 784))   # Tensor.training is False here
preds = out.argmax(axis=-1).numpy()
print("Predictions:", preds)

# Save
safe_save(get_state_dict(model), "simple_mlp.safetensors")
```
"""


# ── Main generation ────────────────────────────────────────────────────────────

def generate_level(level: int):
    out_dir = OUT_BASE / f"level_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process directive files
    for rel_path in DIRECTIVE_FILES:
        content = process_file(rel_path, level)
        # Determine output path (flatten tensor/ subdir into filename)
        out_name = rel_path.replace("/", "_")
        out_file = out_dir / out_name
        out_file.write_text(content)
        print(f"  Wrote {out_file.relative_to(OUT_BASE)}")

    # Include quickstart.md for levels 3-5 as-is
    if level >= 3:
        qs_src = DOCS_DIR / QUICKSTART_FILE
        qs_dst = out_dir / QUICKSTART_FILE
        qs_dst.write_text(qs_src.read_text())
        print(f"  Wrote {qs_dst.relative_to(OUT_BASE)}")

    # Level 5 extras
    if level == 5:
        si_file = out_dir / "special_instructions.md"
        si_file.write_text(SPECIAL_INSTRUCTIONS_MD)
        print(f"  Wrote {si_file.relative_to(OUT_BASE)}")

        pm_file = out_dir / "pytorch_migration.md"
        pm_file.write_text(PYTORCH_MIGRATION_MD)
        print(f"  Wrote {pm_file.relative_to(OUT_BASE)}")


def main():
    print(f"Generating documentation levels in {OUT_BASE}")
    for level in range(1, 6):
        print(f"\n=== Level {level} ===")
        generate_level(level)
    print("\nDone.")


if __name__ == "__main__":
    main()
