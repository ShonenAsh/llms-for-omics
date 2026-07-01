"""Single-shot LLM code output parsing for r-polars benchmarks.

Adapted from tinygrad_test/generate.py. Supports R task stubs with docs
injected as comments.
"""
import re
import json
from pathlib import Path

import instructor
from instructor import Mode
from instructor.core.exceptions import IncompleteOutputException
import litellm
from pydantic import BaseModel


# Prompt models

class SystemPrompt(BaseModel):
    content: str

class TaskPrompt(BaseModel):
    content: str

class Docs(BaseModel):
    content: str

class PriorAttempts(BaseModel):
    content: str


# Output model

class Solution(BaseModel):
    reasoning: str
    code: str


# Helpers
# Strip fences is not needed when using guided_json that enforces json output
# But for some vllm configs, guided_json might fail, hence this is a failsafe
def strip_fences(code: str) -> str:
    """Strip markdown code fences (python or r) if the model included them."""
    m = re.search(r"```(?:python|r)?\s*(.*?)```", code, re.DOTALL)
    return m.group(1).strip() if m else code.strip()


def load_prior_attempts(context_dir: Path) -> PriorAttempts | None:
    """Load previously generated task files as context."""
    parts = [
        f"### {p.name}\n\n```r\n{p.read_text()}\n```"
        for p in sorted(context_dir.glob("task_*.R"))
    ]
    return PriorAttempts(content="\n\n".join(parts)) if parts else None


def load_docs(docs_dir: Path) -> Docs:
    """Concatenate all markdown/ text files in docs_dir."""
    parts = [
        f"# {p.stem}\n\n{p.read_text()}"
        for p in sorted(docs_dir.glob("*"))
        if p.is_file() and p.suffix != ".py"
    ]
    return Docs(content="\n\n---\n\n".join(parts))

# ZTo investigate failures
def log_usage(usage, tag: str = "OK") -> None:
    if usage is None:
        print(f"[{tag}] no usage available")
        return
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    print(f"[{tag}] prompt={usage.prompt_tokens} "
          f"completion={usage.completion_tokens} "
          f"total={usage.total_tokens}"
          + (f" reasoning={reasoning}" if reasoning else ""))


# Generation

def generate(
    system: SystemPrompt,
    task: TaskPrompt,
    docs: Docs,
    model: str = "claude-opus-4-6",
    prior: PriorAttempts | None = None,
    api_base: str | None = None,
    max_tokens: int = 32000,
    extra_body: dict | None = None,
) -> Solution:
    client = instructor.from_litellm(litellm.completion, mode=Mode.JSON)

    parts = []
    if docs.content:
        parts.append(f"## Documentation\n\n{docs.content}")
    if prior:
        parts.append(f"## Your Previous Implementations\n\n{prior.content}")
    parts.append(f"## Task\n\n{task.content}")
    user_content = "\n\n---\n\n".join(parts)

    guided = {"guided_json": Solution.model_json_schema()}
    if extra_body:
        extra_body = {**extra_body, **guided}
    else:
        extra_body = guided

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system.content},
            {"role": "user",   "content": user_content},
        ],
        response_model=Solution,
    )
    if api_base is not None:
        kwargs["api_base"] = api_base
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    try:
        solution, completion = client.chat.completions.create_with_completion(**kwargs)
        log_usage(completion.usage, tag="OK")
    except IncompleteOutputException as e:
        log_usage(getattr(e.last_completion, "usage", None), tag="INCOMPLETE")
        raise

    solution.code = strip_fences(solution.code)
    return solution


# Entry point

if __name__ == "__main__":
    import argparse
    import os
    litellm.num_retries = 2
    litellm.request_timeout = 1200
    parser = argparse.ArgumentParser(description="Generate a task solution via LLM.")
    parser.add_argument("--task",        required=True,   help="Path to task_*.R stub")
    parser.add_argument("--docs",        required=True,   help="Path to docs directory")
    parser.add_argument("--prompt",      required=True,   help="Path to Prompt.md")
    parser.add_argument("--output",      required=True,   help="Where to write the solution .R")
    parser.add_argument("--model",       default=os.environ.get("MODEL", "claude-opus-4-6"))
    parser.add_argument("--context-dir", default=None,    help="Directory of prior task solutions to inject as context")
    parser.add_argument("--api-base",    default=None,    help="Base URL for OpenAI-compatible API endpoint")
    parser.add_argument("--max-tokens",  type=int,        default=32000, help="Max tokens for generation (default: 32000)")
    parser.add_argument("--extra-body",  type=json.loads, default=None,  help='JSON string e.g. \'{"chat_template_kwargs": {"thinking_budget": 8192}}\'')
    args = parser.parse_args()

    system = SystemPrompt(content=Path(args.prompt).read_text())
    task   = TaskPrompt(content=Path(args.task).read_text())
    docs   = load_docs(Path(args.docs)) if Path(args.docs).exists() else Docs(content="")
    prior  = load_prior_attempts(Path(args.context_dir)) if args.context_dir else None

    solution = generate(
        system, task, docs,
        model=args.model,
        prior=prior,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        extra_body=args.extra_body,
    )
    Path(args.output).write_text(solution.code)
    print(f"Written → {args.output}")
