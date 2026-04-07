# LLM Coding Benchmark

Evaluates LLM coding ability on [tinygrad](https://github.com/tinygrad/tinygrad) tasks.
Each task in `tasks/` is a Python stub — the LLM fills in the implementation.
Submissions are scored automatically against the test suite in `tests/`.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python benchmark.py <submission_dir> [<submission_dir> ...]
```
