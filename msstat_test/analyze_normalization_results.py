#!/usr/bin/env python3
"""Parse METRIC lines logged by tests/test_06_choose_normalization.R out of
results.md files across doc conditions/runs, and tabulate FDP/log2FC-bias
evidence per condition per contrast.

This is the "prove it with results, not vibes" deliverable for the
normalization-choice tier: it does not judge correctness itself (that would
reintroduce an LLM-judge-style vibe check) -- it only aggregates the numbers
tests/test_06_choose_normalization.R already computed against known spike-in
ground truth (see reference/probe_normalization_output.txt).

Usage:
  python analyze_normalization_results.py submissions/
"""
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRIC_RE = re.compile(
    r"METRIC contrast=(?P<contrast>\S+) chosen_normalization=(?P<norm>.*?) "
    r"FDP=(?P<fdp>[\d.]+|NA) n_ecoli_sig=(?P<n_ecoli>\d+) n_human_sig=(?P<n_human>\d+) "
    r"median_ecoli_log2FC=(?P<ecoli_fc>-?[\d.]+) expected_ecoli_log2FC=(?P<exp_fc>-?[\d.]+) "
    r"median_human_log2FC=(?P<human_fc>-?[\d.]+)"
)

CONTRASTS = ["E-A", "D-A", "C-A", "B-A"]


def parse_results_md(path):
    rows = []
    text = path.read_text(errors="replace")
    for m in METRIC_RE.finditer(text):
        d = m.groupdict()
        rows.append({
            "contrast": d["contrast"],
            "normalization": d["norm"].strip(),
            "fdp": float(d["fdp"]) if d["fdp"] != "NA" else None,
            "n_ecoli_sig": int(d["n_ecoli"]),
            "n_human_sig": int(d["n_human"]),
            "median_ecoli_log2FC": float(d["ecoli_fc"]),
            "expected_ecoli_log2FC": float(d["exp_fc"]),
            "median_human_log2FC": float(d["human_fc"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions_dir",
                     help="Path to submissions/ (contains <condition>/run_*/results.md)")
    args = ap.parse_args()
    root = Path(args.submissions_dir)

    # condition -> contrast -> [metric rows]
    data = defaultdict(lambda: defaultdict(list))
    norm_choices = defaultdict(lambda: defaultdict(int))

    for results_file in sorted(root.glob("*/run_*/results.md")):
        condition = results_file.parent.parent.name
        for row in parse_results_md(results_file):
            data[condition][row["contrast"]].append(row)
            norm_choices[condition][row["normalization"]] += 1

    if not data:
        print(f"No METRIC lines found under {root} -- "
              "has the normalization-choice tier (task_06) been run yet?")
        return

    for condition in sorted(data):
        print(f"\n=== condition: {condition} ===")
        print("  normalization choices (static grep of submitted code):")
        for norm, count in sorted(norm_choices[condition].items(), key=lambda kv: -kv[1]):
            print(f"    {count:>2}x  {norm}")

        print(f"\n  {'contrast':10} {'n':>3} {'mean FDP':>9} {'median FDP':>11} "
              f"{'mean ecoli log2FC':>18} {'expected':>9} {'mean human log2FC':>18}")
        for contrast in CONTRASTS:
            rows = data[condition].get(contrast, [])
            if not rows:
                continue
            fdps = [r["fdp"] for r in rows if r["fdp"] is not None]
            ecoli_fcs = [r["median_ecoli_log2FC"] for r in rows]
            human_fcs = [r["median_human_log2FC"] for r in rows]
            expected = rows[0]["expected_ecoli_log2FC"]
            mean_fdp = statistics.mean(fdps) if fdps else float("nan")
            median_fdp = statistics.median(fdps) if fdps else float("nan")
            print(f"  {contrast:10} {len(rows):>3} "
                  f"{mean_fdp:>9.3f} {median_fdp:>11.3f} "
                  f"{statistics.mean(ecoli_fcs):>18.3f} {expected:>9.3f} "
                  f"{statistics.mean(human_fcs):>18.3f}")

    conditions = sorted(data)
    if len(conditions) > 1 and "pitfall_note" in conditions:
        print("\n=== headline delta: pitfall_note vs other conditions (E-A contrast) ===")
        pn_fdps = [r["fdp"] for r in data["pitfall_note"].get("E-A", []) if r["fdp"] is not None]
        for condition in conditions:
            if condition == "pitfall_note":
                continue
            other_fdps = [r["fdp"] for r in data[condition].get("E-A", []) if r["fdp"] is not None]
            if pn_fdps and other_fdps:
                print(f"  {condition:16} mean FDP {statistics.mean(other_fdps):.3f}  "
                      f"vs  pitfall_note mean FDP {statistics.mean(pn_fdps):.3f}")


if __name__ == "__main__":
    main()
