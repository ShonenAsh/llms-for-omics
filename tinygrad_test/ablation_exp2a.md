# Ablation Study: Documentation Level vs Pass Rate (exp_2a)

## Experimental Setup

- **Models:** `big-pickle` and `deepseek-v4-flash-free` via OpenCode Zen API
- **Documentation levels:**
  - Level 1: API signatures only
  - Level 2: signatures + descriptions
  - Level 3: Level 2 + first code example + quickstart.md
  - Level 4: Level 2 + all code examples with executed output + quickstart.md
  - Level 5: Level 4 + special_instructions.md + pytorch_migration.md
- **Tasks:** 10 tinygrad coding tasks per level per model (63 tests)
- **Context:** prior attempt injection (each task sees previously generated solutions)
- **API:** Zen API via litellm, model IDs `openai/big-pickle` and `openai/deepseek-v4-flash-free`

---

## Results by Level

| Model | L1 | L2 | L3 | L4 | L5 |
|-------|:--:|:--:|:--:|:--:|:--:|
| big-pickle | 53/63 (84.1%) | 52/63 (82.5%) | **58/63 (92.1%)** | 57/63 (90.5%) | 54/63 (85.7%) |
| deepseek | 49/63 (77.8%) | 52/63 (82.5%) | 50/63 (79.4%) | 57/63 (90.5%) | **58/63 (92.1%)** |

Neither model shows monotonic improvement with more documentation. Each peaks at a different level:
- big-pickle peaks at L3 (92.1%), then regresses at L4-L5
- deepseek peaks at L5 (92.1%), with a dip at L3

---

## Failure Categorization

Each failure falls into one of three categories:

### Hallucinations (API does not exist or wrong parameter name)

| Hallucination | big-pickle | deepseek | Appears at |
|---|---|---|---|
| `from tinygrad.jit import TinyJit` | L1, L2 | L1, L2 | L1-L2 (lower levels) |
| `Tensor.cat(..., dims=1)` instead of `dim=` | L1 | -- | L1 |
| `with Tensor.no_grad():` (does not exist) | -- | L3 | L3 |
| `.clip(min=0)` instead of `min_=0` | -- | L1, L3 | L1, L3 |
| `Tensor.kaiming_uniform(...)` (does not exist) | -- | L4, L5 | L4-L5 (higher levels) |
| `x.linear(...)` (does not exist on Tensor) | -- | L5 | L5 |

Key pattern: deepseek produces more hallucinations (6 types vs 2), and higher-level docs trigger NEW hallucinations (`kaiming_uniform`, `x.linear`) rather than preventing them.

### Logic Errors (correct API, wrong algorithm)

| Logic Error | big-pickle | deepseek | Root Cause |
|---|---|---|---|
| Missing `Tensor.train()` before optimizer | L1, L2, L3, L4, L5 | L1, L2, L3, L4 | `optim.step()` fails without `Tensor.training = True` |
| Causal mask produces NaN | L1, L4, L5 | L2, L3, L4 | `ones(T,T).triu(1) * -inf` = `0 * -inf` = NaN |
| ResidualBlock naming off | L1, L4 | L1, L2, L4, L5 | Test expects `fc1`/`fc2`, model uses `w1`/`w2` or `linear1`/`linear2` |
| LayerNorm weight not trainable | -- | L3 | `Tensor.ones(...)` without `requires_grad=True` |

### Style (unused imports)

Unused imports increase at higher levels. L5 shows a spike for both models:

- big-pickle L5: unused `dtypes`, `np`, `get_parameters`, `annotations`, `List`, `Tuple` across 7 files
- deepseek L5: unused `SGD`, `List`, `Tuple`, `dtypes`, `np` across 5 files

---

## Detailed Hallucination Analysis

### H1: Wrong JIT import path (`tinygrad.jit`)

The correct import is `from tinygrad import TinyJit`. The deprecated `tinygrad.jit` submodule does not exist in tinygrad 0.12.0.

Appears at L1-L2 for both models. Fixed from L3 onward (once docs include example code showing the correct import). This is the one case where more documentation clearly helps.

Files affected:
- `big_pickle_exp2a_lvl1/task_06_training_loop.py:4` -- `from tinygrad.jit import TinyJit`
- `deepseek_exp2a_lvl1/task_06_training_loop.py:3` -- `from tinygrad.jit import TinyJit`
- Same pattern in L2 for both models

### H2: `dims=` vs `dim=` on Tensor.cat

Tinygrad uses `dim=` keyword. The model writes `dims=`, likely from PyTorch convention where both `dim` and `dims` sometimes appear.

Only big-pickle L1:
- `big_pickle_exp2a_lvl1/task_10_transformer.py:123` -- `idx.cat((), dims=1)`

### H3: `Tensor.no_grad()` (non-existent)

Tinygrad 0.12.0 does not have `Tensor.no_grad()`. The correct approach is `Tensor.detach()` or setting `Tensor.training = False`.

Only deepseek L3:
- `deepseek_exp2a_lvl3/task_02_linear_regression.py:39` -- `with Tensor.no_grad():`

This confirms `Tensor.no_grad()` hallucination originates from the model's training data (PyTorch bleeding), not from our docs (since we removed `no_grad` from the docs before this run).

### H4: `.clip(min=0)` instead of `min_=0`

Tinygrad uses `min_` (trailing underscore avoids clashing with Python builtin). Models persistently use `min`, the PyTorch convention.

Only deepseek:
- `deepseek_exp2a_lvl1/task_09_custom_losses.py:73` -- `margin_diff.clip(min=0)`
- `deepseek_exp2a_lvl3/task_09_custom_losses.py:53` -- `(margin - D).clip(min=0)`

big-pickle avoids this by using alternative patterns (positional args, `.maximum()`, `.where()`).

### H5: `Tensor.kaiming_uniform(...)` (non-existent)

Tinygrad does not have `Tensor.kaiming_uniform`. The correct API is `Tensor.kaiming_uniform_()`. This hallucination appears only at higher doc levels (L4, L5) for deepseek, suggesting the PyTorch migration guide's weight init patterns triggered it.

Files affected:
- `deepseek_exp2a_lvl4/task_07_custom_layers.py:46,49` -- appears twice
- `deepseek_exp2a_lvl5/task_07_custom_layers.py:14,30,45,47` -- appears 4 times

### H6: `x.linear(...)` as instance method

Tinygrad has `nn.Linear` as a module, not `Tensor.linear()` as an instance method. The model invented `x.linear(weight, bias)` as a functional call.

Only deepseek L5:
- `deepseek_exp2a_lvl5/task_07_custom_layers.py:53,55`

---

## Logic Error Analysis

### E1: Missing `Tensor.train()` before optimizer step (most common failure)

The optimizer raises `RuntimeError: Tensor.training=False, Tensor.training must be enabled`. This affects 5/6 model-level combinations. The only run that fully avoids it is deepseek L5, which correctly uses `with Tensor.train():`.

Models attempt various patterns:
- `optim.step()` without any training context -- most common, always fails
- `Tensor.training = True` (manual flag) -- appears in deepseek L2, works but not idiomatic
- `with Tensor.train():` -- correct, appears sporadically

This is the single largest source of failures. Missing training context causes `test_3d_training_reduces_loss` to fail in almost every run, and cascades into `test_6c_training_decreases_loss` and `test_10e_training_decreases_loss`.

### E2: Causal mask NaN (0 * -inf)

The pattern `Tensor.ones(T, T).triu(1) * float('-inf')` produces NaN because IEEE 754 `0 * -inf = NaN`. When the attention weights go through softmax, NaN propagates to the output and kills training.

This affects big-pickle in 3/5 levels. Deepseek sometimes gets it right (L1, L5) by using `Tensor.full()` or `where()` patterns.

### E3: ResidualBlock attribute naming

The test expects `model.fc1` and `model.fc2` to exist as `nn.Linear` modules. Models use various names:
- `w1`, `w2` (big-pickle L1)
- `linear1`, `linear2` (deepseek L1, L2)
- `fc1_weight`, `fc1_bias` as raw tensors (deepseek L4, L5)
- `ff1`, `ff2` (big-pickle L5)

Consistently correct naming never appears. `test_7c_residual_connection_present` fails in 8/10 model-level runs.

---

## Summary

| Finding | Verdict |
|---------|---------|
| More documentation improves pass rates? | No -- peak at different levels per model |
| Fixes structured bugs (import paths)? | Yes -- JIT path fixed from L3 onward for both models |
| Teaches API-level conventions (min_, dim, cat)? | No -- these persist across all doc levels |
| Higher docs introduce NEW hallucinations? | Yes -- `kaiming_uniform` and `x.linear` appear only at L4-L5 |
| Deepseek vs big-pickle hallucination rate | Deepseek: 6 hallucination types. Big-pickle: 2 types |
| Most consistent failure | `Tensor.training=False` before optimizer -- affects 5/6 model-level combos |
| Only test that improved with docs | JIT import path (L1-L2 wrong, L3-L5 correct) |

**Conclusion:** Documentation helps fix specific import/library usage bugs (H1) but does not teach API-level quirks (min_ vs min, dim vs dims, training mode). Higher doc levels (L4-L5) introduce new hallucinations through over-generalization of PyTorch migration patterns. The optimal level is model-dependent: L3 for big-pickle (58/63), L5 for deepseek (58/63). The `Tensor.training` problem remains the single largest source of failures regardless of documentation level.
