# LLM Coding Benchmark

Evaluates LLM coding ability on [tinygrad](https://github.com/tinygrad/tinygrad) tasks.
Each task in `tasks/` is a Python stub — the LLM fills in the implementation.
Submissions are scored automatically against the test suite in `tests/`.

## Experiments

### Experiment 1: Baseline (no docs)
LLMs generate code from memory alone. No documentation is provided.

### Experiment 2a: Documentation Ablation (additive)
Tests how progressively richer documentation affects LLM performance.
Currently, I am only testing single-shot code generation (No agents), but in the future
there are plans of implementing a thin yet practical agentic system for improved control by LLMs.

Each level adds to the previous:

| Level | Contents |
|-------|----------|
| `level_1` | API signatures + type hints |
| `level_2` | + descriptions |
| `level_3` | + one code example per method |
| `level_4` | + all code examples with executed output |
| `level_5` | + special instructions (lazy execution) + PyTorch migration guide |

Docs for each level live in `exp_2a/level_N/`. The system prompt is in `exp_2a/Prompt.md`.

## Docker

Each experiment/level is a separate image built from a shared base. Docs and the system
prompt are baked in at build time. Model and API key are injected at run time.

Build all images:
```bash
bash build_images.sh
```

### Running benchmarks

The `MODEL` env var supports any [litellm model string](https://docs.litellm.ai/docs/providers).
Use `API_BASE` to route to a custom OpenAI-compatible endpoint.

```bash
# Claude (direct)
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e MODEL=claude-opus-4-6 \
  tinygrad-bench:exp_2a-level_3

# GPT-4o (direct)
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e MODEL=gpt-4o \
  tinygrad-bench:exp_2a-level_3

# Gemini (direct)
docker run --rm \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e MODEL=gemini/gemini-2.0-flash \
  tinygrad-bench:exp_2a-level_3

# Hugging Face (free, rate-limited)
docker run --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL=huggingface/Qwen/Qwen2.5-Coder-32B-Instruct \
  tinygrad-bench:exp_2a-level_1

# Custom OpenAI-compatible endpoint (e.g., local vLLM, OpenCode Zen)
docker run --rm \
  -e MODEL=openai/gpt-4 \
  -e API_BASE=https://your-endpoint.com/v1 \
  -e OPENAI_API_KEY=sk-your_key \
  tinygrad-bench:exp_2a-level_3
```

## Local setup

```bash
uv sync
```

Generate solutions for a given task, docs level, and model:
```bash
python generate.py \
  --task tasks/task_01_tensor_basics.py \
  --docs exp_2a/level_3 \
  --prompt exp_2a/Prompt.md \
  --output submissions/task_01_tensor_basics.py \
  --model claude-opus-4-6
```

Once all submissions are gathered, score them:
```bash
uv run pytest tests/ -v --import-mode=importlib
```

Or use `benchmark.py` to run against a submissions folder directly:
```bash
uv run python benchmark.py submissions/
```
