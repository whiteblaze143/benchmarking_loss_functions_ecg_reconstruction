#!/usr/bin/env python3
"""Build cross-paper comparison matrix from fold8 evaluation results.

Usage:
    python build_comparison_matrix.py --results-dir outputs/cross_paper_evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--baseline-paper", type=int, default=2, help="Baseline paper for delta comparison")
    args = parser.parse_args()

    all_results = {}
    for f in sorted(args.results_dir.glob("paper*_fold8.json")):
        key = f.stem.replace("_fold8", "")
        all_results[key] = json.loads(f.read_text())

    if not all_results:
        print("No results found!")
        return

    # Get baseline
    baseline_key = f"paper{args.baseline_paper:02d}"
    baseline_auroc = None
    if baseline_key in all_results and "error" not in all_results[baseline_key]:
        baseline_auroc = all_results[baseline_key]["best_macro_auroc"]

    # Build table
    rows = []
    for paper_key in sorted(all_results.keys()):
        r = all_results[paper_key]
        if "error" in r:
            rows.append({
                "paper": paper_key,
                "best_variant": "ERROR",
                "auroc": None,
                "delta": None,
                "n_variants": 0,
                "status": r["error"][:50],
            })
        else:
            auroc = r["best_macro_auroc"]
            delta = (auroc - baseline_auroc) if baseline_auroc is not None else None
            rows.append({
                "paper": paper_key,
                "best_variant": r["best_variant"],
                "auroc": auroc,
                "delta": delta,
                "n_variants": r["n_variants_evaluated"],
                "status": "BEAT BASELINE" if delta and delta > 0.01 else "NULL" if delta and abs(delta) <= 0.01 else "WORSE" if delta and delta < -0.01 else "—",
            })

    # Print markdown table
    print(f"\n# Cross-Paper Comparison (Fold 8 AUROC)")
    print(f"\nBaseline: {baseline_key} (AUROC={baseline_auroc:.4f})" if baseline_auroc else "\nBaseline: N/A")
    print()
    print(f"| Paper | Best Variant | AUROC | Δ vs Baseline | Variants | Status |")
    print(f"|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(x["auroc"] or 0)):
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "—"
        delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
        print(f"| {r['paper']} | {r['best_variant']} | {auroc_str} | {delta_str} | {r['n_variants']} | {r['status']} |")

    # Save as markdown
    md_path = args.results_dir / "comparison_matrix.md"
    lines = [
        "# Cross-Paper Comparison Matrix (Fold 8 AUROC)",
        "",
        f"Baseline: {baseline_key} (AUROC={baseline_auroc:.4f})" if baseline_auroc else "Baseline: N/A",
        "",
        "| Paper | Best Variant | AUROC | Δ vs Baseline | # Variants | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -(x["auroc"] or 0)):
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "—"
        delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
        lines.append(f"| {r['paper']} | {r['best_variant']} | {auroc_str} | {delta_str} | {r['n_variants']} | {r['status']} |")

    lines.extend([
        "",
        "## Interpretation",
        "- **BEAT BASELINE**: Δ > +0.01 — genuine novelty candidate",
        "- **NULL**: |Δ| ≤ 0.01 — informative negative (like P14)",
        "- **WORSE**: Δ < -0.01 — representation hurts diagnosis",
        "- **ERROR**: Training or evaluation failed",
    ])
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to: {md_path}")


if __name__ == "__main__":
    main()
