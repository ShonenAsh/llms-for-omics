"""Single-shot LLM code output parsing for benchmarks."""
import re
from pathlib import Path

import instructor
import litellm
from pydantic import BaseModel


# ── Prompt models ──────────────────────────────────────────────────────────────

class SystemPrompt(BaseModel):
    content: str

class TaskPrompt(BaseModel):
    content: str

class Docs(BaseModel):
    content: str

class PriorAttempts(BaseModel):
    content: str


# ── Output model ───────────────────────────────────────────────────────────────

class Solution(BaseModel):
    reasoning: str
    code: str  # Python only


# ── Helpers ────────────────────────────────────────────────────────────────────

def strip_fences(code: str) -> str:
    """Strip markdown python fences if the model included them."""
    m = re.search(r"```python\s*(.*?)```", code, re.DOTALL)
    return m.group(1).strip() if m else code.strip()


def load_prior_attempts(context_dir: Path) -> PriorAttempts | None:
    """Load previously generated task files as context."""
    parts = [
        f"### {p.name}\n\n```python\n{p.read_text()}\n```"
        for p in sorted(context_dir.glob("task_*.py"))
    ]
    return PriorAttempts(content="\n\n".join(parts)) if parts else None


def load_docs(docs_dir: Path) -> Docs:
    """Concatenate all markdown files in docs_dir into a single Docs object."""
    parts = [
        f"# {p.stem}\n\n{p.read_text()}"
        for p in sorted(docs_dir.glob("*.md"))
    ]
    return Docs(content="\n\n---\n\n".join(parts))


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(
    system: SystemPrompt,
    task: TaskPrompt,
    docs: Docs,
    model: str = "claude-opus-4-6",
    prior: PriorAttempts | None = None,
) -> Solution:
    client = instructor.from_litellm(litellm.completion)

    user_content = f"## Documentation\n\n{docs.content}"
    if prior:
        user_content += f"\n\n---\n\n## Your Previous Implementations\n\n{prior.content}"
    user_content += f"\n\n---\n\n## Task\n\n{task.content}"

    solution: Solution = client.chat.completions.create(
        model=model,
        max_tokens=8096,
        messages=[
            {"role": "system", "content": system.content},
            {"role": "user",   "content": user_content},
        ],
        response_model=Solution,
    )
    solution.code = strip_fences(solution.code)
    return solution





# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate a task solution via LLM.")
    parser.add_argument("--task",   required=True, help="Path to task_*.py stub")
    parser.add_argument("--docs",   required=True, help="Path to docs directory")
    parser.add_argument("--prompt", required=True, help="Path to Prompt.md")
    parser.add_argument("--output",      required=True, help="Where to write the solution .py")
    parser.add_argument("--model",       default=os.environ.get("MODEL", "claude-opus-4-6"))
    parser.add_argument("--context-dir", default=None,  help="Directory of prior task solutions to inject as context")
    args = parser.parse_args()

    system = SystemPrompt(content=Path(args.prompt).read_text())
    task   = TaskPrompt(content=Path(args.task).read_text())
    docs   = load_docs(Path(args.docs))
    prior  = load_prior_attempts(Path(args.context_dir)) if args.context_dir else None

    solution = generate(system, task, docs, model=args.model, prior=prior)
    Path(args.output).write_text(solution.code)
    print(f"Written → {args.output}")
