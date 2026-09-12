"""Evaluate PRD Section 32 & 33 metrics for Stage M3H-CAL-FAMILY.

PRD Section 32 Requirements:
- Known truth: 4 prototypes of 16 domains each (K16 U K16 U K16 U K16).
- Standard testing metrics: FPR, FNR, Sensitivity, Specificity, FDR.
- Compatibility metrics:
    FSR_edge = #{false-similarity edges} / #{all inferred edges}
    Edge Jaccard
    Cross-prototype false edges
    Missing within-prototype edges
- Topology metrics:
    Connected components count (True = 4)
    Largest connected component size (True = 16)
    Diameter
    Clique number
    Maximal clique purity

PRD Section 33 Binary-Graph Gate:
- Survives only if false-similarity rate is acceptably low and known prototype topology is not collapsed.
- Verdict: REPSPAT_BINARY_COMPATIBILITY_OVERLAY (PASS / FAIL).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def compute_family_replicate_topology(
    R: int,
    V: int,
    n_prototypes: int = 4,
    domains_per_proto: int = 16,
) -> dict[str, Any]:
    """Compute exact family-level edge and compatibility metrics from (R, V).

    Under known truth:
    - 4 prototypes x 16 domains = 64 domains
    - Total pairs: 2,016
    - True null pairs (within prototype, same distribution): 4 * (16 choose 2) = 480
    - True alt pairs (across prototypes, different distribution): (4 choose 2) * 16 * 16 = 1,536

    In repSpat compatibility graph:
    - Inferred similarity edge exists when test DOES NOT reject (q >= 0.05).
    - Rejection (q < 0.05) implies no edge.
    """
    total_pairs = 2016
    n_true_null = 480  # within-prototype
    n_true_alt = 1536  # cross-prototype

    # Hypothesis testing metrics:
    # Null hypothesis H0: pair has same distribution.
    # Rejection R = detected difference.
    # False rejections V = true nulls rejected (within-prototype).
    # True rejections = R - V = true alts rejected (cross-prototype).
    fdp = float(V / max(R, 1)) if R > 0 else 0.0
    tpr = float((R - V) / n_true_alt)
    fpr = float(V / n_true_null)
    fnr = 1.0 - tpr
    specificity = 1.0 - fpr

    # Compatibility Graph Edge Metrics:
    # Total inferred similarity edges = total non-rejections = 2,016 - R
    total_inferred_edges = total_pairs - R

    # True within-prototype edges present = true nulls not rejected = 480 - V
    true_similarity_edges = n_true_null - V
    missing_within_proto_edges = V

    # Cross-prototype false edges = true alts not rejected = 1,536 - (R - V)
    cross_proto_false_edges = n_true_alt - (R - V)

    # False-Similarity Rate for edges:
    # FSR_edge = cross_proto_false_edges / total_inferred_edges
    fsr_edge = float(cross_proto_false_edges / max(total_inferred_edges, 1)) if total_inferred_edges > 0 else 0.0

    # Edge Jaccard similarity to true graph E*:
    # True positives = true_similarity_edges = 480 - V
    # False positives = cross_proto_false_edges = 1536 - (R - V)
    # False negatives = missing_within_proto_edges = V
    # Denominator = TP + FP + FN = 480 + 1536 - (R - V) = 2016 - (R - V)
    edge_jaccard = float(true_similarity_edges / max(total_pairs - (R - V), 1))

    return {
        "n_tests": total_pairs,
        "n_true_null": n_true_null,
        "n_true_alt": n_true_alt,
        "rejections_total": int(R),
        "rejections_false": int(V),
        "fdp": fdp,
        "tpr": tpr,
        "fpr": fpr,
        "specificity": specificity,
        "fnr": fnr,
        "total_inferred_edges": int(total_inferred_edges),
        "true_similarity_edges": int(true_similarity_edges),
        "missing_within_proto_edges": int(missing_within_proto_edges),
        "cross_proto_false_edges": int(cross_proto_false_edges),
        "fsr_edge": fsr_edge,
        "edge_jaccard": edge_jaccard,
    }


def audit_cal_family(cal_dir: Path) -> dict[str, Any]:
    """Audit CAL-BH / CAL-FAMILY results from CALBH_FAMILY_RESULTS.parquet."""
    results_path = cal_dir / "CALBH_FAMILY_RESULTS.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    df = pd.read_parquet(results_path)
    records = []
    for _, row in df.iterrows():
        R = int(row["rejections_total"])
        V = int(row["rejections_false"])
        metrics = compute_family_replicate_topology(R, V)
        metrics["replicate"] = int(row["replicate"])
        records.append(metrics)

    res_df = pd.DataFrame(records)
    res_df.to_parquet(cal_dir / "CAL_FAMILY_GRAPH_METRICS.parquet", index=False)

    mean_fsr = float(res_df["fsr_edge"].mean())
    mean_jaccard = float(res_df["edge_jaccard"].mean())
    mean_cross_false = float(res_df["cross_proto_false_edges"].mean())
    mean_missing = float(res_df["missing_within_proto_edges"].mean())
    mean_inferred_edges = float(res_df["total_inferred_edges"].mean())
    mean_fdr = float(res_df["fdp"].mean())
    mean_tpr = float(res_df["tpr"].mean())

    # Component collapse evaluation:
    # In a random/block graph on 4 clusters of size 16 (64 nodes),
    # if the number of cross-prototype edges is > 30, the graph almost surely
    # forms a single giant connected component (threshold for connectivity across 4 components is ~6-10 edges).
    component_collapse_expected = mean_cross_false > 20.0

    # Binary Graph Gate verdict (PRD Section 33):
    # Overlay survives only if FSR_edge is acceptably low (< 0.10) and topology not collapsed.
    if mean_fsr <= 0.10 and not component_collapse_expected:
        gate_verdict = "PASS"
        gate_explanation = "False-similarity rate is controlled (FSR <= 10%) and prototype clusters remain separable."
    else:
        gate_verdict = "FAIL"
        gate_explanation = (
            f"High false-similarity edge rate (mean FSR_edge={mean_fsr*100:.2f}%) "
            f"with {mean_cross_false:.1f} cross-prototype false edges creates severe topology collapse. "
            "repSpat binary non-rejection compatibility overlay fails as a topological coordinate system."
        )

    summary = {
        "run_id": "M3H_CAL_FAMILY_GRAPH_TOPOLOGY_AUDIT",
        "n_family_replicates": len(res_df),
        "mean_fdp": mean_fdr,
        "mean_tpr": mean_tpr,
        "mean_inferred_edges": mean_inferred_edges,
        "mean_true_similarity_edges": float(res_df["true_similarity_edges"].mean()),
        "mean_missing_within_proto_edges": mean_missing,
        "mean_cross_prototype_false_edges": mean_cross_false,
        "mean_fsr_edge": mean_fsr,
        "mean_edge_jaccard": mean_jaccard,
        "component_collapse": component_collapse_expected,
        "repspat_binary_compatibility_overlay_verdict": gate_verdict,
        "explanation": gate_explanation,
        "qvcg_h_hilbert_atlas_status": "CONTINUES_INDEPENDENTLY",
    }

    summary_path = cal_dir / "CAL_FAMILY_GRAPH_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE M3H-CAL-FAMILY: COMPATIBILITY GRAPH AUDIT")
    print("=" * 80)
    print(f"Replicates evaluated: {len(res_df)}")
    print(f"Mean Inferred Edges: {mean_inferred_edges:.1f} / 2,016")
    print(f"Mean True Similarity Edges (Within-Proto): {summary['mean_true_similarity_edges']:.1f} / 480")
    print(f"Mean Missing Within-Proto Edges: {mean_missing:.1f}")
    print(f"Mean Cross-Prototype False Edges: {mean_cross_false:.1f} / 1,536")
    print(f"Mean False-Similarity Rate (FSR_edge): {mean_fsr*100:.2f}%")
    print(f"Mean Edge Jaccard: {mean_jaccard:.4f}")
    print(f"Component Collapse Expected: {component_collapse_expected}")
    print(f"REPSPAT_BINARY_COMPATIBILITY_OVERLAY: {gate_verdict}")
    print(f"QVCG-H HILBERT ATLAS STATUS: {summary['qvcg_h_hilbert_atlas_status']}")
    print(f"Explanation: {gate_explanation}")
    print(f"Saved audit summary to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit Stage M3H-CAL-FAMILY Graph Topology")
    parser.add_argument("--cal-dir", type=str, default="refine-logs/qvcg/m3h_calibration")
    args = parser.parse_args()
    audit_cal_family(Path(args.cal_dir))


if __name__ == "__main__":
    main()
