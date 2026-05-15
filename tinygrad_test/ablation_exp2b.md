# Ablation Study: Factor Isolation (exp_2b)

## Design

Starting from "Full Docs" (exp_2a Level 5), selectively remove ONE documentation factor per ablation:

| Ablation | Removed | Rationale |
|----------|---------|-----------|
| 1a | All type annotations from signatures | Does knowing param types help or confuse? |
| 1b | Return type annotations only | Are return shapes/shape hints important? |
| 2 | All code example blocks + outputs | Do code examples reduce hallucinations? |
| 3 | special_instructions.md | Does the lazy execution/Tensor.train() doc help? |
| 4 | pytorch_migration.md | Does the PyTorch equivalence guide help or cause confusion? |

---

## Pass Rate Comparison

### Baseline (exp_2a L5)
- big-pickle: 54/63 (85.7%)
- deepseek: 58/63 (92.1%)

### big-pickle

| Variant | Passed | Failed | Rate | vs Baseline |
|---------|:-----:|:-----:|:----:|:-----------:|
| Baseline (L5) | 54 | 9 | 85.7% | -- |
| 1a (no types) | 52 | 11 | 82.5% | -2 |
| 1b (no returns) | 49 | 14 | 77.8% | -5 |
| **2 (no examples)** | **59** | **4** | **93.7%** | **+5** |
| 3 (no special) | 50 | 13 | 79.4% | -4 |
| 4 (no migration) | 54 | 9 | 85.7% | 0 |

### deepseek-v4-flash-free

| Variant | Passed | Failed | Rate | vs Baseline |
|---------|:-----:|:-----:|:----:|:-----------:|
| Baseline (L5) | 58 | 5 | 92.1% | -- |
| 1a (no types) | 54 | 9 | 85.7% | -4 |
| **1b (no returns)** | **59** | **4** | **93.7%** | **+1** |
| 2 (no examples) | 52 | 11 | 82.5% | -6 |
| 3 (no special) | 50 | 13 | 79.4% | -8 |
| 4 (no migration) | 56 | 7 | 88.9% | -2 |

---

## Factor-by-Factor Analysis

### Factor 1a: Type Annotations (all types removed)

**Finding: Removing types hurts both models.**

big-pickle drops from 85.7% to 82.5% (-2). deepseek drops from 92.1% to 85.7% (-4).

big-pickle regressions introduced by removing types:
- Focal loss now uses `.clamp(min=1e-7)` (TypeError: `min` should be `min_`)
- Contrastive loss uses `.clamp(min=0)` (same kwarg error)
- Without type annotations showing `min_=None` in the clamp signature, the model defaults to PyTorch convention `min=`

deepseek regressions:
- MLP class output shape wrong (no return type `-> Tensor` to guide shape)
- Cross-entropy computation broken
- LayerNorm weight `requires_grad` missing
- Unused imports increase (9 failures, vs 5 at baseline)

Type annotations serve as parameter-name hints. Without them, both models regressed to PyTorch defaults for parameter names.

### Factor 1b: Return Type Annotations Only

**Finding: Split effect. Helps deepseek (+1), hurts big-pickle (-5).**

This is big-pickle's worst result. The specific regressions:
- `Tensor.sqrt(d_k)` appears in task_05 attention (AttributeError: 'int' object has no attribute 'cast'). The model calls `Tensor.sqrt(d_k)` as a class method on an integer.
- `TinyModel` class is referenced in function signatures but never defined (NameError).  
- SDPA weights computation is broken (cascades from sqrt error).
- Contrastive loss uses `.clamp(min=0)`.
- Unused imports increase across multiple files.

Without return type annotations (`-> Tensor`, `-> tuple[Tensor, Tensor]`), big-pickle loses shape context and invents wrong API patterns. deepseek improves slightly, suggesting deepseek can infer shapes from descriptions alone.

### Factor 2: Code Examples

**Finding: Opposite effect. Helps big-pickle (+5), hurts deepseek (-6).**

big-pickle's best result at 93.7% (59/63). Only 4 failures:
- `test_3d_training_reduces_loss` (missing `Tensor.train()`)
- `test_6c_training_returns_correct_length` / `test_6c_training_decreases_loss` (training loop)
- `test_7c_residual_connection_present` (fc1/fc2 naming)

Without examples, big-pickle:
- Does not hallucinate `Tensor.sqrt(d_k)` (present in 1b)
- Does not use `from tinygrad.jit import TinyJit` (present in L1-L2)
- No focal loss errors (present in 1a)
- Training loop and training mode remain the only consistent failures

deepseek without examples drops to 82.5%:
- JIT import regresses to `from tinygrad.jit import TinyJit`
- Model state tests fail (TinyModel class import error)
- Training loop LR schedule fails
- Without code examples showing `TinyJit` usage and model state patterns, deepseek falls back to hallucinated imports

### Factor 3: Special Instructions (Lazy Execution, Tensor.training, JIT, no_grad)

**Finding: Removing special instructions hurts both models.**

big-pickle drops to 79.4% (-4). deepseek drops to 79.4% (-8) -- worst result for both.

big-pickle regressions:
- Cosine LR schedule tests fail (test_6b) -- model can't implement proper schedule without reference
- Training loop tests fail (test_6c)
- Unused imports spike (6 files with unused imports)

deepseek regressions:
- All training loop tests fail (both cosine LR and loss decrease)
- Causal mask fails (test_5b) -- future positions not masked
- Transformer generation fails with `clip(min=0)` (wrong kwarg) and `clip(max=...)` (wrong kwarg)
- Contrastive loss uses `clip(min=0)` -- same kwarg error
- Unused imports increase

The special_instructions.md contained examples of `Tensor.train()` context, JIT usage, and lazy execution patterns. Without it, both models struggle with training loops and tinygrad-specific API quirks. deepseek is hit harder because it relied on the doc's explicit patterns for `Tensor.train()` and generation.

### Factor 4: PyTorch Migration Guide

**Finding: big-pickle unchanged (0), deepseek hurt (-2).**

big-pickle remains at 85.7% (identical to baseline). Removing the migration guide neither helps nor hurts.

deepseek drops from 92.1% to 88.9% (-2):
- Training loop tests fail (test_6c)
- Transformer generation fails (test_10d)
- Residual block naming fails
- Unused imports in task_01 and task_08

deepseek was using patterns from the migration guide (train loop examples, save/load patterns). Without it, some of these regress.

---

## Hallucination Tracking

| Hallucination | Baseline | 1a | 1b | 2 | 3 | 4 |
|--------------|:--------:|:--:|:--:|:--:|:--:|:--:|
| **big-pickle** | | | | | | |
| `Tensor.sqrt(d_k)` | absent | absent | present | absent | absent | absent |
| `tinygrad.jit` import | absent | absent | absent | absent | present | present |
| `clamp(min=)` | absent | present | present | absent | absent | absent |
| `TinyModel` undefined | absent | absent | present | absent | absent | absent |
| **deepseek** | | | | | | |
| `tinygrad.jit` import | absent | absent | absent | present | absent | absent |
| `clamp(min=)` / `clip(min=)` | absent | absent | absent | present | present | absent |
| `clip(max=)` | absent | absent | absent | absent | present | absent |
| `Tensor.no_grad()` | absent | absent | absent | absent | absent | absent |

Note: `Tensor.no_grad()` hallucination is absent in all ablations (removed from docs before this run), confirming it originated from the documentation and not the model's training data.

---

## Conclusions

### What documentation characteristics matter?

**There is no single characteristic that universally reduces hallucinations. The effect is entirely model-dependent.**

| Characteristic | big-pickle | deepseek |
|---|---|---|
| Type annotations | Slightly helpful (prevents `clamp(min=)`) | Helpful (prevents wrong param names, shapes) |
| Return type annotations | Harmful (triggers `Tensor.sqrt(d_k)`) | Slightly helpful (shape hints) |
| Code examples | Harmful (triggers over-copying bad patterns) | Essential (prevents wrong imports, missing class defs) |
| Special instructions | Helpful (train loop, cosine LR) | Essential (train loop, generation, clip params) |
| PyTorch migration guide | No effect | Slightly helpful (train loop patterns) |

### Key observations

1. **big-pickle's best result is WITHOUT code examples (93.7%).** The model over-copies from examples and generalizes poorly. When examples are removed, it relies on API descriptions and produces cleaner code.

2. **deepseek's best result is WITHOUT return type annotations (93.7%).** Return types shape hints are not needed. deepseek infers shapes from descriptions alone.

3. **deepseek's worst result is WITHOUT special instructions (79.4%).** The special_instructions.md provides critical context for training loops, lazy execution, and the `Tensor.train()` API. Without it, deepseek regresses badly.

4. **Return type annotations trigger `Tensor.sqrt(d_k)` in big-pickle.** The `-> Tensor` return annotation combined with the `Tensor.sqrt()` signature format makes big-pickle interpret `sqrt` as a class-level method call. This hallucination only appears in ablation 1b (no return types) where type stripping changes the signature format.

5. **The `clamp(min=)` hallucination is documentation-dependent.** When type annotations are present (baseline), both models avoid this error. When types are removed (1a, 1b), `min=` reappears. The documentation acts as a parameter-name reference.

6. **Unused imports increase when documentation is reduced.** Both models import more unused symbols when documentation factors are removed, suggesting they cast a wider net without knowing what's actually needed.
