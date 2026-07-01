"""
Aggregate per-test results across all experiments and runs into a single CSV.

Supports both pytest (Python) and testthat (R) output formats.

Expects layout:
    submissions/
        <exp_tag>/
            run_00/results.md
            run_01/results.md
            ...
        <exp_tag>/
            ...

Output CSV columns:
    experiment, test, c, n, pass_rate, pass@1, pass@5, pass@10


Usage: (args are optional)
    parse_results.py <SUBMISSION_DIR> --out <OUTPUT.csv>

"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from math import comb

# Matches lines like: tests/test_task_01.py::test_add_tensors PASSED
PYTEST_LINE = re.compile(
    r"^(?P<nodeid>\S+::test_\w+)\s+(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)",
    re.MULTILINE,
)

# Matches lines like: test_01_basic.R::create_mtcars_df PASSED
R_LINE = re.compile(
    r"^(?P<nodeid>\S+\.R::\w+)\s+(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)",
    re.MULTILINE,
)

K_VALUES = (1, 5, 10)

def calculate_pass_at_k(n: int, c: int, ks=(1, 5, 10)) -> dict[str, float]:
    """Return {'pass@k': value} for each k in ks."""
    out = {}
    for k in ks:
        if k > n:
            out[f"pass@{k}"] = float("nan")  # undefined for k > n
        elif n - c < k:
            out[f"pass@{k}"] = 1.0
        else:
            out[f"pass@{k}"] = 1.0 - comb(n - c, k) / comb(n, k)
    return out

def parse_results(results_md: Path) -> dict[str, bool]:
    """Return {test_function_name: passed} from pytest or testthat output."""
    text = results_md.read_text()
    raw: dict[str, bool] = {}

    for pattern in (PYTEST_LINE, R_LINE):
        for m in pattern.finditer(text):
            raw[m.group("nodeid")] = m.group("outcome") in ("PASSED", "XPASS")

    # Shorten to bare function name where unambiguous
    by_fname: dict[str, list[str]] = defaultdict(list)
    for nodeid in raw:
        by_fname[nodeid.split("::")[-1]].append(nodeid)

    results: dict[str, bool] = {}
    for fname, nodeids in by_fname.items():
        if len(nodeids) == 1:
            results[fname] = raw[nodeids[0]]
        else:
            for nid in nodeids:
                results[nid] = raw[nid]
    return results


def aggregate_experiment(exp_dir: Path) -> tuple[dict[str, int], int]:
    """
    Walk all run_*/results.md under exp_dir.
    Returns ({test_name: pass_count}, n_runs).
    """
    run_dirs = sorted(exp_dir.glob("run_*"))
    counts: dict[str, int] = defaultdict(int)
    all_tests: set[str] = set()

    for run_dir in run_dirs:
        results_md = run_dir / "results.md"
        if not results_md.exists():
            print(f"  WARN: {results_md} missing", file=sys.stderr)
            continue
        per_run = parse_results(results_md)
        all_tests.update(per_run.keys())
        for test, passed in per_run.items():
            if passed:
                counts[test] += 1

    # Ensure every test seen anywhere has an entry (0 if never passed)
    for test in all_tests:
        counts.setdefault(test, 0)

    return dict(counts), len(run_dirs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("submissions", type=Path, nargs="?", default=Path("submissions"),
                   help="Root submissions directory (default: ./submissions)")
    p.add_argument("--out", type=Path, default=Path("pass_k_results.csv"),
                   help="Output CSV path (default: ./pass_k_results.csv)")
    args = p.parse_args()

    root = args.submissions.resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    rows = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if not any(exp_dir.glob("run_*")):
            continue  # not an experiment dir

        print(f"Processing {exp_dir.name} ...")
        counts, n = aggregate_experiment(exp_dir)
        for test in sorted(counts):
            c = counts[test]
            rows.append({
                "experiment": exp_dir.name,
                "test":       test,
                "c":          c,
                "n":          n,
                "pass_rate":  f"{c/n:.4f}" if n else "0",
                **calculate_pass_at_k(n, c, K_VALUES),
            })

    if not rows:
        print("No experiment data found.", file=sys.stderr)
        return 1

    fieldnames = ["experiment", "test", "c", "n", "pass_rate"] + [f"pass@{k}" for k in K_VALUES]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n→ {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())