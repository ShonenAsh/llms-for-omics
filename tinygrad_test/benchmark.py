#!/usr/bin/env python3
"""
Usage
-----
    python benchmark.py <submission_dir> [<submission_dir> ...]

Each submission directory must contain the completed task_*.py files.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"


def run_tests(test_file: Path, submission_dir: Path) -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(submission_dir) + (f":{existing}" if existing else "")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header", "--import-mode=importlib"],
        capture_output=True, text=True, env=env, timeout=180,
    )
    duration = round(time.perf_counter() - t0, 2)

    out = proc.stdout
    return {
        "passed":   out.count(" PASSED"),
        "failed":   out.count(" FAILED"),
        "errors":   out.count(" ERROR"),
        "duration": duration,
    }


def score(submission_dir: Path) -> None:
    submission_dir = submission_dir.resolve()
    test_files = sorted(TESTS_DIR.glob("test_*.py"))

    total_passed, total_tests = 0, 0
    for tf in test_files:
        r = run_tests(tf, submission_dir)
        total = r["passed"] + r["failed"] + r["errors"]
        total_passed += r["passed"]
        total_tests  += total
        print(f"  {tf.stem}  {r['passed']}/{total}  [{r['duration']}s]")

    pct = round(total_passed / total_tests * 100, 1) if total_tests else 0
    print(f"  {'─' * 50}")
    print(f"  TOTAL  {total_passed}/{total_tests}  ({pct}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+", help="Submission directories to evaluate")
    args = parser.parse_args()

    for sub in args.submissions:
        print(f"\n{sub}")
        score(Path(sub))


if __name__ == "__main__":
    main()
