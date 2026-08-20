#!/usr/bin/env python3
"""Parse METRIC lines logged by tests/test_05_choose_normalization_talus.R out of
results.md files across doc conditions/runs, and tabulate the SE-based evidence
for the Talus chromatin-proteomics tier.

Unlike analyze_normalization_results.py (FDP-based, for the spike-in benchmark,
where the correct answer is `normalization = FALSE`), this dataset has no
species-label ground truth -- the diagnostic here (verified in
reference/probe_talus_normalization_output.txt) is whether turning off
normalization inflates standard error (SE) on known landmark proteins
(BRD2/3/4, the direct target of the dBET6 degrader) without meaningfully
shifting their fold-change estimates. Reference values: mean SE across all
proteins is ~0.155 with equalizeMedians (recommended here) vs ~0.246 with
normalization = FALSE.

Usage:
  python analyze_talus_results.py submissions/
"""
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

MEAN_SE_RE = re.compile(
    r"METRIC contrast=(?P<contrast>\S+) chosen_normalization=(?P<norm>.*?) "
    r"mean_SE_all=(?P<mean_se>[\d.]+)"
)
PROTEIN_RE = re.compile(
    r"METRIC contrast=(?P<contrast>\S+) chosen_normalization=(?P<norm>.*?) "
    r"protein=(?P<protein>\S+) log2FC=(?P<log2fc>-?[\d.]+) SE=(?P<se>[\d.]+) "
    r"adj\.pvalue=(?P<adjp>[\d.]+)"
)

LANDMARK_PROTEINS = ["BRD2_HUMAN", "BRD3_HUMAN", "BRD4_HUMAN"]

# Pinned reference values -- reference/probe_talus_normalization_output.txt
REFERENCE_MEAN_SE = {"equalizeMedians": 0.1554, "FALSE": 0.2464}


def parse_results_md(path):
    mean_rows, protein_rows = [], []
    text = path.read_text(errors="replace")
    for m in MEAN_SE_RE.finditer(text):
        d = m.groupdict()
        mean_rows.append({"contrast": d["contrast"], "normalization": d["norm"].strip(),
                           "mean_se": float(d["mean_se"])})
    for m in PROTEIN_RE.finditer(text):
        d = m.groupdict()
        protein_rows.append({"contrast": d["contrast"], "normalization": d["norm"].strip(),
                              "protein": d["protein"], "log2FC": float(d["log2fc"]),
                              "SE": float(d["se"]), "adj_pvalue": float(d["adjp"])})
    return mean_rows, protein_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions_dir",
                     help="Path to submissions/ (contains <condition>/run_*/results.md)")
    args = ap.parse_args()
    root = Path(args.submissions_dir)

    mean_by_cond = defaultdict(list)
    protein_by_cond = defaultdict(lambda: defaultdict(list))

    for results_file in sorted(root.glob("*/run_*/results.md")):
        condition = results_file.parent.parent.name
        mean_rows, protein_rows = parse_results_md(results_file)
        mean_by_cond[condition].extend(mean_rows)
        for r in protein_rows:
            protein_by_cond[condition][r["protein"]].append(r)

    if not mean_by_cond:
        print(f"No METRIC lines found under {root} -- has task_05 been run yet?")
        return

    print(f"Reference (pinned, see reference/probe_talus_normalization_output.txt): "
          f"mean SE across all proteins is ~{REFERENCE_MEAN_SE['equalizeMedians']} with "
          f"equalizeMedians (recommended) vs ~{REFERENCE_MEAN_SE['FALSE']} with "
          f"normalization = FALSE. Closer to the first number is the correct outcome here.")

    for condition in sorted(mean_by_cond):
        print(f"\n=== condition: {condition} ===")
        norm_counts = defaultdict(int)
        for r in mean_by_cond[condition]:
            norm_counts[r["normalization"]] += 1
        print("  normalization choices (static grep of submitted code):")
        for norm, count in sorted(norm_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>2}x  {norm}")

        mses = [r["mean_se"] for r in mean_by_cond[condition]]
        print(f"\n  mean_SE_all across all proteins: mean={statistics.mean(mses):.4f}, "
              f"n={len(mses)}")

        print(f"\n  {'protein':12} {'n':>3} {'mean log2FC':>12} {'mean SE':>9}")
        for protein in LANDMARK_PROTEINS:
            rows = protein_by_cond[condition].get(protein, [])
            if not rows:
                continue
            fcs = [r["log2FC"] for r in rows]
            ses = [r["SE"] for r in rows]
            print(f"  {protein:12} {len(rows):>3} {statistics.mean(fcs):>12.4f} "
                  f"{statistics.mean(ses):>9.4f}")


if __name__ == "__main__":
    main()
