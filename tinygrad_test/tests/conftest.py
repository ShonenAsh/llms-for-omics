"""
Shared fixtures and utilities for the tinygrad LLM benchmark suite.
"""
import ast
import sys
from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# AST-based unused import checker
# ---------------------------------------------------------------------------

def _collect_imports(tree: ast.Module) -> dict[str, ast.stmt]:
    """Return {bound_name: node} for every top-level import in the file."""
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import foo.bar` binds name `foo`; `import foo as f` binds `f`
                bound = alias.asname or alias.name.split(".")[0]
                imports[bound] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue  # star imports are untrackable
                bound = alias.asname or alias.name
                imports[bound] = node
    return imports


def _collect_used_names(tree: ast.Module) -> set[str]:
    """
    Return every Name id that appears outside of import statements.
    Also captures the root name of attribute chains (e.g. `nn` in `nn.Linear`).
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Walk down to the root Name of an attribute chain
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    return used


def get_unused_imports(path: Path) -> list[str]:
    """
    Parse *path* with the AST and return the list of imported names that are
    never referenced anywhere in the file body.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []  # syntax errors are caught by other tests
    imports = _collect_imports(tree)
    used    = _collect_used_names(tree)
    return sorted(name for name in imports if name not in used)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _find_task_file(task_stem: str) -> Path | None:
    """Search sys.path for <task_stem>.py (the submission file)."""
    for entry in sys.path:
        candidate = Path(entry) / f"{task_stem}.py"
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def task_source(request) -> Path:
    """
    Resolve the submitted task_*.py file that corresponds to this test module.
    test_01_tensor_basics  →  task_01_tensor_basics.py
    """
    test_stem = Path(request.fspath).stem          # e.g. test_01_tensor_basics
    task_stem = "task" + test_stem[4:]             # e.g. task_01_tensor_basics
    path = _find_task_file(task_stem)
    if path is None:
        pytest.skip(f"{task_stem}.py not found in sys.path")
    return path


@pytest.fixture(scope="module")
def unused_imports(task_source: Path) -> list[str]:
    """List of imported names in the submission file that are never used."""
    return get_unused_imports(task_source)
