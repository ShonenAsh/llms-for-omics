"""
generate_ablations.py
=====================
Generates 5 ablation documentation folders for exp_2b.

Each ablation starts from the "Full Docs" (Level 5 of exp_2a) and selectively
removes one documentation factor:

  1a — no_types:       strip ALL type annotations from API signatures
  1b — no_return_shapes: strip only return-type annotations from signatures
  2  — no_examples:    remove all code example blocks (```python … ```) + outputs
  3  — no_special:     remove special_instructions.md
  4  — no_migration:   remove pytorch_migration.md
"""

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "exp_2a" / "level_5"         # Full Docs source
OUT = HERE / "exp_2b"

DOC_FILES = [
    "dtypes.md",
    "nn.md",
    "quickstart.md",
    "special_instructions.md",
    "pytorch_migration.md",
    "tensor_creation.md",
    "tensor_elementwise.md",
    "tensor_index.md",
    "tensor_movement.md",
    "tensor_ops.md",
    "tensor_properties.md",
]


# ---------------------------------------------------------------------------
# Post-processing callables  (each returns (file_path -> content | None))
# ---------------------------------------------------------------------------

def strip_type_annotations(content: str) -> str:
    """Remove every `: …` param annotation and ` -> ReturnType` from #### lines."""
    def _clean_sig(line: str) -> str:
        # Remove quoted param type annotations  `: '…'`
        line = re.sub(r": '([^']*)'", "", line)
        # Remove unquoted param type annotations  `: typename`
        line = re.sub(r": ([^,)=]+)(?=[,)=])", "", line)
        # Remove return type annotation  ` -> …`
        line = re.sub(r" -> .+$", "", line)
        return line

    lines = content.splitlines(keepends=True)
    return "".join(_clean_sig(l) if l.startswith("#### ") else l for l in lines)


def strip_return_annotations(content: str) -> str:
    """Remove only ` -> ReturnType` from #### signature lines."""
    def _clean_sig(line: str) -> str:
        line = re.sub(r" -> .+$", "", line)
        return line

    lines = content.splitlines(keepends=True)
    return "".join(_clean_sig(l) if l.startswith("#### ") else l for l in lines)


_CODE_BLOCK_RE = re.compile(
    r"```python[^\n]*\n.*?```\n"       # python fenced block
    r"(?:[ \t]*\n)*"                   # optional blank lines
    r"(?:```\n.*?```\n[ \t]*\n?)?",    # optional output block
    re.DOTALL,
)

def strip_code_examples(content: str) -> str:
    """Remove all ```python … ``` blocks and their trailing output blocks."""
    return _CODE_BLOCK_RE.sub("", content)


def strip_file(filename: str):
    """Return a callable that removes `filename` from the output directory."""
    def _remove(existing: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in existing.items() if k != filename}
    return _remove


ABLATIONS = {
    "ablation_1a_no_types":         strip_type_annotations,
    "ablation_1b_no_return_shapes": strip_return_annotations,
    "ablation_2_no_examples":       strip_code_examples,
    "ablation_3_no_special":        strip_file("special_instructions.md"),
    "ablation_4_no_migration":      strip_file("pytorch_migration.md"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for ablation_name, transform in ABLATIONS.items():
        out_dir = OUT / ablation_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Read all source files into {name: content}
        files: dict[str, str] = {}
        for fname in DOC_FILES:
            src_path = SRC / fname
            if src_path.exists():
                files[fname] = src_path.read_text()

        # Apply the ablation transformation
        # (strip_file returns a dict-level filter; others return str content)
        if callable(transform) and transform.__name__ == "_remove":
            files = transform(files)
        else:
            transformed: dict[str, str] = {}
            for fname, content in files.items():
                transformed[fname] = transform(content)
            files = transformed

        # Write out
        for fname, content in files.items():
            (out_dir / fname).write_text(content)

        count = len(files)
        print(f"  {ablation_name:30s}  {count} files")

    # Copy Prompt.md
    prompt_src = HERE / "exp_2a" / "Prompt.md"
    if prompt_src.exists():
        shutil.copy2(prompt_src, OUT / "Prompt.md")
        print(f"  Copied Prompt.md")


if __name__ == "__main__":
    main()
