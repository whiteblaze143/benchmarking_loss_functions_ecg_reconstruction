#!/usr/bin/env python3
"""Generate publication-quality 4-panel figure for Milestone R024.

Panels:
A: Canonical & Reduced Lead Benchmark Comparison (AUROC with 95% CIs)
B: Limb Lead Geometry & Derived Leads Benchmark (AUROC with 95% CIs)
C: Continuous Oblique Vector Generalization (S^7 Span)
D: Training Dynamics & Learning Trajectories (Val AUROC across epochs)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_r024_benchmark_curves")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
RESULTS_FILE = CODEX_DIR / "paper07_operator/r024_benchmark_results/r024_benchmark_results.json"
OUTPUT_PNG = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/r024_operator_span_benchmark_curves.png")


def _get_val(results, cfg, model):
    """Return (mean_auroc, lo_err, hi_err) — falls back to raw metrics when no bootstrap."""
    boot = results[cfg].get("bootstrap", {}).get("bootstrap_cis", {})
    if model in boot:
        m = boot[model]["mean"]
        lo = m - boot[model]["ci_95"][0]
        hi = boot[model]["ci_95"][1] - m
        return m, lo, hi
    # Fallback: use raw metrics, zero error bars
    m = results[cfg]["metrics"][model]["macro_auroc"]
    return m, 0.0, 0.0


def _has_model(results, cfg, model):
    """Return True if model data is present for a config."""
    return (
        model in results[cfg].get("bootstrap", {}).get("bootstrap_cis", {})
        or model in results[cfg].get("metrics", {})
    )


def main():
    logger.info("Loading benchmark results...")
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Load training histories if available
    hist_basis = []
    hist_span = []
    p_basis = MODELS_DIR / "basis_only/summary.json"
    p_span = MODELS_DIR / "operator_span/summary.json"
    if p_basis.exists():
        with open(p_basis) as f:
            hist_basis = json.load(f).get("history", [])
    if p_span.exists():
        with open(p_span) as f:
            hist_span = json.load(f).get("history", [])

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    color_canon = "#1f77b4"  # Blue
    color_basis = "#ff7f0e"  # Orange
    color_span = "#2ca02c"   # Green
    color_graph = "#d62728"  # Red

    # -------------------------------------------------------------
    # Panel A: Canonical & Reduced Lead Benchmark Comparison
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    configs_a = ["Q8_full", "S6_precordial", "S3_reduced", "S2_reduced", "S1_smartwatch"]
    labels_a = ["Q8 Full\n(8 leads)", "S6 Precordial\n(V1-V6)", "S3 Reduced\n(I, II, V2)", "S2 Reduced\n(I, II)", "S1 Watch\n(Lead I)"]
    x = np.arange(len(configs_a))
    width = 0.2

    # GraphECG
    graphedge_vals = [results[c].get("graphecg_benchmark", np.nan) for c in configs_a]
    ax_a.bar(x - 1.5 * width, graphedge_vals, width, label="GraphECG (SOTA)", color=color_graph, alpha=0.85, edgecolor="black", linewidth=0.8)

    # Canonical
    canon_means, canon_lo, canon_hi = zip(*[_get_val(results, c, "canonical") for c in configs_a])
    ax_a.bar(x - 0.5 * width, canon_means, width, yerr=[canon_lo, canon_hi], capsize=3, label="Canonical SetOperator", color=color_canon, alpha=0.85, edgecolor="black", linewidth=0.8)

    # Basis-Only
    if _has_model(results, configs_a[0], "basis_only"):
        basis_means, basis_lo, basis_hi = zip(*[_get_val(results, c, "basis_only") for c in configs_a])
        ax_a.bar(x + 0.5 * width, basis_means, width, yerr=[basis_lo, basis_hi], capsize=3, label="Basis-Only Training", color=color_basis, alpha=0.85, edgecolor="black", linewidth=0.8)

    # Operator-Span
    if _has_model(results, configs_a[0], "operator_span"):
        span_means, span_lo, span_hi = zip(*[_get_val(results, c, "operator_span") for c in configs_a])
        ax_a.bar(x + 1.5 * width, span_means, width, yerr=[span_lo, span_hi], capsize=3, label="Operator-Span Training", color=color_span, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels_a, fontsize=9.5)
    ax_a.set_ylabel("Macro AUROC", fontsize=11, fontweight="bold")
    ax_a.set_ylim(0.70, 0.96)
    ax_a.set_title("A. Canonical & Reduced Clinical Lead Configurations", fontsize=12, fontweight="bold", pad=10)
    ax_a.legend(loc="lower right", frameon=True, fontsize=9)
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel B: Limb Lead Geometry & Derived Leads Benchmark
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    configs_b = ["S6_limb", "Derived_limb_only", "Lead_III_single", "aVF_single"]
    labels_b = ["6 Limb Leads\n(I, II, III, aVR-F)", "Derived Limb Only\n(III, aVR, aVL, aVF)", "Single Lead III\n(Derived)", "Single Lead aVF\n(Derived)"]
    x_b = np.arange(len(configs_b))

    graphedge_b = [results[c].get("graphecg_benchmark", np.nan) for c in configs_b]
    ax_b.bar(x_b - 1.5 * width, graphedge_b, width, label="GraphECG (SOTA)", color=color_graph, alpha=0.85, edgecolor="black", linewidth=0.8)

    canon_b, canon_b_lo, canon_b_hi = zip(*[_get_val(results, c, "canonical") for c in configs_b])
    ax_b.bar(x_b - 0.5 * width, canon_b, width, yerr=[canon_b_lo, canon_b_hi], capsize=3, label="Canonical SetOperator", color=color_canon, alpha=0.85, edgecolor="black", linewidth=0.8)

    if _has_model(results, configs_b[0], "basis_only"):
        basis_b, basis_b_lo, basis_b_hi = zip(*[_get_val(results, c, "basis_only") for c in configs_b])
        ax_b.bar(x_b + 0.5 * width, basis_b, width, yerr=[basis_b_lo, basis_b_hi], capsize=3, label="Basis-Only Training", color=color_basis, alpha=0.85, edgecolor="black", linewidth=0.8)

    if _has_model(results, configs_b[0], "operator_span"):
        span_b, span_b_lo, span_b_hi = zip(*[_get_val(results, c, "operator_span") for c in configs_b])
        ax_b.bar(x_b + 1.5 * width, span_b, width, yerr=[span_b_lo, span_b_hi], capsize=3, label="Operator-Span Training", color=color_span, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, fontsize=9.5)
    ax_b.set_ylabel("Macro AUROC", fontsize=11, fontweight="bold")
    ax_b.set_ylim(0.70, 0.92)
    ax_b.set_title("B. Limb Lead Geometry & Derived Linear Combinations", fontsize=12, fontweight="bold", pad=10)
    ax_b.legend(loc="lower right", frameon=True, fontsize=9)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel C: Continuous Oblique Vector Generalization
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    configs_c = ["S_icm", "S_icm_plus_leadI", "Test_obliques_3", "All_16_leads"]
    labels_c = ["S_icm (V3 - V2)\nSingle Oblique", "S_icm + Lead I\n(2 Oblique/Basis)", "3 Test Obliques\n(Random in S^7)", "All 16 Leads\n(Full Ensemble)"]
    x_c = np.arange(len(configs_c))

    graphedge_c = [results[c].get("graphecg_benchmark", np.nan) for c in configs_c]
    ax_c.bar(x_c - 1.5 * width, graphedge_c, width, label="GraphECG (SOTA)", color=color_graph, alpha=0.85, edgecolor="black", linewidth=0.8)

    canon_c, canon_c_lo, canon_c_hi = zip(*[_get_val(results, c, "canonical") for c in configs_c])
    ax_c.bar(x_c - 0.5 * width, canon_c, width, yerr=[canon_c_lo, canon_c_hi], capsize=3, label="Canonical SetOperator", color=color_canon, alpha=0.85, edgecolor="black", linewidth=0.8)

    if _has_model(results, configs_c[0], "basis_only"):
        basis_c, basis_c_lo, basis_c_hi = zip(*[_get_val(results, c, "basis_only") for c in configs_c])
        ax_c.bar(x_c + 0.5 * width, basis_c, width, yerr=[basis_c_lo, basis_c_hi], capsize=3, label="Basis-Only Training", color=color_basis, alpha=0.85, edgecolor="black", linewidth=0.8)

    if _has_model(results, configs_c[0], "operator_span"):
        span_c, span_c_lo, span_c_hi = zip(*[_get_val(results, c, "operator_span") for c in configs_c])
        ax_c.bar(x_c + 1.5 * width, span_c, width, yerr=[span_c_lo, span_c_hi], capsize=3, label="Operator-Span Training", color=color_span, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(labels_c, fontsize=9.5)
    ax_c.set_ylabel("Macro AUROC", fontsize=11, fontweight="bold")
    ax_c.set_ylim(0.65, 0.96)
    ax_c.set_title("C. Continuous Oblique Vector Generalization (S^7 Space)", fontsize=12, fontweight="bold", pad=10)
    ax_c.legend(loc="lower right", frameon=True, fontsize=9)
    ax_c.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel D: Validation AUROC Learning Dynamics
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    if hist_basis:
        ep_b = [h["epoch"] for h in hist_basis]
        val_b = [h["val_macro_auroc"] for h in hist_basis]
        ax_d.plot(ep_b, val_b, "-o", color=color_basis, label="Basis-Only Training", markersize=4, linewidth=1.8)
    if hist_span:
        ep_s = [h["epoch"] for h in hist_span]
        val_s = [h["val_macro_auroc"] for h in hist_span]
        ax_d.plot(ep_s, val_s, "-s", color=color_span, label="Operator-Span Training", markersize=4, linewidth=1.8)

    ax_d.axhline(0.9309, color=color_canon, linestyle="--", linewidth=1.5, label="Canonical Frozen (0.9309)")
    ax_d.axhline(0.9128, color=color_graph, linestyle=":", linewidth=1.5, label="GraphECG Q8 Baseline (0.9128)")
    ax_d.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("Validation Macro AUROC (Q8)", fontsize=11, fontweight="bold")
    ax_d.set_title("D. Training Trajectories & Convergence", fontsize=12, fontweight="bold", pad=10)
    ax_d.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax_d.grid(True, linestyle="--", alpha=0.5)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    logger.info(f"Saved publication figure to {OUTPUT_PNG}.")


if __name__ == "__main__":
    main()
