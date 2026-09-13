"""Patient-level diagnostic analysis of Gate 3B cohort heterogeneity."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RESULT_DIR = "tit_ecg/results/empirical_multi_dataset"
PAIR_PATH = os.path.join(RESULT_DIR, "gate3b_pair_diagnostics.csv")
PREDICTORS = [
    "min_cluster_size",
    "m_star",
    "G_star",
    "temporal_separation_ms",
    "heart_rate_bpm",
    "morphology_distance",
]


def patient_bootstrap_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    recurrence = pairs[pairs["pair_kind"] == "recurrence_control"].copy()
    positive = pairs[pairs["pair_kind"] == "positive_control"].copy()

    patient = recurrence.groupby(["dataset", "patient_id"], as_index=False).agg(
        n_recurrence_control_pairs=("rejected_by_graph_rule", "size"),
        recurrence_rejections=("rejected_by_graph_rule", "sum"),
        recurrence_rejection_rate=("rejected_by_graph_rule", "mean"),
        phase=("phase_1", "first"),
        median_cluster_size=("min_cluster_size", "median"),
        m_star=("m_star", "first"),
        G_star=("G_star", "first"),
        median_temporal_separation_ms=("temporal_separation_ms", "median"),
        heart_rate_bpm=("heart_rate_bpm", "first"),
        median_morphology_distance=("morphology_distance", "median"),
    )
    patient.to_csv(os.path.join(RESULT_DIR, "gate3b_patient_diagnostics.csv"), index=False)

    dataset_rows = []
    for dataset, group in patient.groupby("dataset"):
        lo, hi = patient_bootstrap_ci(group["recurrence_rejection_rate"].to_numpy())
        dataset_rows.append({
            "dataset": dataset,
            "n_patients": len(group),
            "n_recurrence_control_pairs": int(group["n_recurrence_control_pairs"].sum()),
            "patient_mean_recurrence_rejection_rate": float(group["recurrence_rejection_rate"].mean()),
            "patient_bootstrap_ci95_low": lo,
            "patient_bootstrap_ci95_high": hi,
            "median_cluster_size": float(group["median_cluster_size"].median()),
            "median_m_star": float(group["m_star"].median()),
            "median_G_star": float(group["G_star"].median()),
            "median_temporal_separation_ms": float(group["median_temporal_separation_ms"].median()),
            "median_heart_rate_bpm": float(group["heart_rate_bpm"].median()),
            "median_morphology_distance": float(group["median_morphology_distance"].median()),
        })
    dataset_summary = pd.DataFrame(dataset_rows)
    dataset_summary.to_csv(os.path.join(RESULT_DIR, "gate3b_dataset_diagnostics.csv"), index=False)

    # Report both raw and within-cohort-rank associations. Their difference
    # distinguishes patient-level trends from dataset-level confounding.
    association_predictors = {
        "cluster_size": "median_cluster_size",
        "m_star": "m_star",
        "G_star": "G_star",
        "temporal_separation": "median_temporal_separation_ms",
        "heart_rate": "heart_rate_bpm",
        "morphology_distance": "median_morphology_distance",
    }
    associations = []
    for name, column in association_predictors.items():
        valid = patient[[column, "recurrence_rejection_rate"]].dropna()
        rho, p_value = spearmanr(valid[column], valid["recurrence_rejection_rate"])
        associations.append({
            "predictor": name,
            "n_patients": len(valid),
            "raw_spearman_rho": float(rho),
            "raw_p_value_exploratory": float(p_value),
            "within_dataset_rank_spearman_rho": float(spearmanr(
                patient.groupby("dataset")[column].rank(pct=True),
                patient["recurrence_rejection_rate"],
            ).statistic),
            "within_dataset_rank_p_value_exploratory": float(spearmanr(
                patient.groupby("dataset")[column].rank(pct=True),
                patient["recurrence_rejection_rate"],
            ).pvalue),
        })
    association_df = pd.DataFrame(associations)
    association_df.to_csv(os.path.join(RESULT_DIR, "gate3b_patient_associations.csv"), index=False)

    positive_summary = []
    for dataset, group in positive.groupby("dataset"):
        positive_summary.append({
            "dataset": dataset,
            "n_positive_control_pairs": len(group),
            "positive_control_rejections": int(group["rejected_by_graph_rule"].sum()),
            "positive_control_power": float(group["rejected_by_graph_rule"].mean()),
        })

    payload = {
        "decision_quantity": "BH-adjusted q_value < 0.05 (same decision used by graph construction)",
        "purity_rule": "exact_phase_and_beat_membership",
        "n_eligible_patients": int(len(patient)),
        "n_recurrence_control_pairs": int(len(recurrence)),
        "overall_patient_mean_recurrence_rejection_rate": float(patient["recurrence_rejection_rate"].mean()),
        "overall_patient_bootstrap_ci95": list(patient_bootstrap_ci(patient["recurrence_rejection_rate"].to_numpy())),
        "overall_pooled_recurrence_rejection_rate": float(recurrence["rejected_by_graph_rule"].mean()),
        "observed_recurrence_phases": sorted(int(x) for x in recurrence["phase_1"].unique()),
        "phase_analysis_limitation": "All admissible recurrence controls are QRS (phase 2); phase heterogeneity is not estimable.",
        "dataset_summary": dataset_rows,
        "patient_level_associations": associations,
        "positive_controls": positive_summary,
        "power_limitation": "Positive controls occur only in Zhejiang; cross-cohort power is not established.",
    }
    with open(os.path.join(RESULT_DIR, "gate3b_heterogeneity_analysis.json"), "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
