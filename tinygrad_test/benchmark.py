#!/usr/bin/env python3
"""
Run the tinygrad benchmark against a submission directory.

Usage:
    python benchmark.py <submission_dir> [--md PATH]

The submission directory must contain the completed task_*.py files.
Pytest output is printed to the terminal and written to
<submission_dir>/results.md (override with --md PATH).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path, help="Directory containing the candidate's task_*.py files")
    p.add_argument("--md", type=Path, help="Where to write the markdown report (default: <submission>/results.md)")
    args = p.parse_args()

    sub = args.submission.resolve()
    md  = args.md.resolve() if args.md else sub / "results.md"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(sub) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS_DIR), "-v"],
        env=env, text=True, capture_output=True,
    )
    output = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr.strip() else "")
    print(output)

    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(output)
    print(f"\n→ {md}")

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
