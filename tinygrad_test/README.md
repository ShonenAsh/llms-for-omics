# tinygrad LLM Coding Benchmark

A fill-in-the-blank coding benchmark that evaluates how well LLMs can use the
[tinygrad](https://github.com/tinygrad/tinygrad) deep-learning framework.

---

## Structure

```
tinygrad_test/
├── tasks/          ← fill-in-the-blank problem files  (give these to the LLM)
├── solutions/      ← reference solutions               (ground truth)
├── tests/          ← pytest test suite                 (auto-grader)
├── benchmark.py    ← runner: scores one or more models
└── mnist.py        ← existing working example
```

---

## Tasks

| # | Task | Difficulty | Key Concepts |
|---|------|-----------|--------------|
| 01 | Tensor Basics | Easy | `Tensor` creation, matmul, reductions, activations, broadcasting |
| 02 | Linear Regression | Easy-Medium | `requires_grad`, forward pass, MSE, manual SGD, `.assign()` |
| 03 | MLP Classifier | Medium | `nn.Linear`, dropout, cross-entropy, accuracy, `Adam` |
| 04 | CNN | Medium | `nn.Conv2d`, `nn.BatchNorm`, pooling, parameter counting |
| 05 | Scaled Dot-Product Attention | Hard | SDPA, causal masking, multi-head attention |
| 06 | JIT Training Loop | Medium-Hard | `TinyJit`, cosine LR schedule, per-epoch metrics |
| 07 | Custom Layers | Medium | LayerNorm, Embedding, residual FFN block |
| 08 | Model Serialization | Medium | `get_state_dict`, `safe_save/load`, `load_state_dict`, freezing |
| 09 | Custom Loss Functions | Medium | Focal loss, Dice loss, Contrastive loss |
| 10 | Transformer Block | Hard | Causal MHA + FFN + LayerNorm + residual, mini language model |

---

## Prompting an LLM

Give the LLM the contents of a `tasks/task_XX_*.py` file and instruct it to
replace every `???` with working tinygrad code.  No other context is needed —
each task file is self-contained with a docstring, imports, and an embedded
auto-grader at the bottom.

**Example system prompt:**

```
You are an expert ML engineer. You will be given a Python file that uses the
tinygrad deep-learning library. Replace every ??? placeholder with correct
tinygrad code. Do not change function signatures, assertions, or imports.
Return only the complete, modified Python file.
```

---

## Running the Benchmark

### Prerequisites

```bash
cd tinygrad_test
uv sync          # or: pip install tinygrad pytest
```

### Verify reference solutions pass

```bash
python benchmark.py --submission solutions/ --model reference
```

### Evaluate a single LLM

Place the LLM's completed files in a directory (e.g. `gpt4o_outputs/`) and run:

```bash
python benchmark.py --submission gpt4o_outputs/ --model gpt-4o
```

### Compare multiple models

```bash
python benchmark.py \
    --submission solutions/        --model reference \
    --submission gpt4o_outputs/    --model gpt-4o \
    --submission claude_outputs/   --model claude-3-7-sonnet
```

Output is printed as a Markdown table and saved to `benchmark_results.json`.

---

## Scoring

Each task is split into multiple pytest test functions.  The score is:

```
score = (tests passed) / (total tests) × 100%
```

Tests are independent — a task can receive partial credit if some sub-parts
are correct and others are not.

### Test count per task

| Task | # Tests |
|------|---------|
| 01 Tensor Basics | 6 |
| 02 Linear Regression | 4 |
| 03 MLP Classifier | 4 |
| 04 CNN | 4 |
| 05 Attention | 5 |
| 06 Training Loop | 5 |
| 07 Custom Layers | 7 |
| 08 Model State | 5 |
| 09 Custom Losses | 7 |
| 10 Transformer | 5 |
| **Total** | **52** |

---

## Running individual tasks directly

Each task file also functions as a standalone script with its own auto-grader:

```bash
cd tinygrad_test
python solutions/task_01_tensor_basics.py
python solutions/task_10_transformer.py
```
