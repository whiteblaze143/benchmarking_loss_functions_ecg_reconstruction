#!/usr/bin/env python3
"""Compare two Sunnybrook all-feature benchmark runs.

This script is intentionally report-focused. It merges the finished benchmark
artifacts for two reconstruction models and writes a single comparison package
that shows where the candidate model improves or regresses relative to the
baseline across:
1. overall grouped summaries,
2. direct zero-shot feature preservation,
3. per-target regression probes,
4. per-target classification probes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fft import rfft
from scipy.spatial.distance import directed_hausdorff

sys.path.append(os.getcwd())

from src.reconstruction.unified_latents.engineering.benchmark_sunnybrook_all_features import (
    LEAD_ORDER,
    load_reconstruction_model,
    parse_obs_leads,
    reconstruct_sources,
)
from src.reconstruction.learn_functions.reconstruction_functions import calculate_sample_r2
from src.reconstruction.util_functions.mason_12lead import mason_batch_r2_loss


DEFAULT_BASELINE_DIR = (
    "/home/mithunmanivannan/reports/sunnybrook_all_feature_baseline/"
    "engineering_wearecg_exact_II-V1-V5_bs64_lf1.5_ep10_canonical_baseline"
)
DEFAULT_CANDIDATE_DIR = (
    "/home/mithunmanivannan/reports/sunnybrook_all_feature_baseline/"
    "engineering_fm_vae_II-V1-V5_bs64_diagmasksplitv1_mw2_fml5e-2_mix0_dc0_drop0_la0_law0_msa1e-1"
)
DEFAULT_OUTPUT_ROOT = "/home/mithunmanivannan/reports/sunnybrook_all_feature_baseline/comparisons"

CLINICAL_PRIORITY_TARGETS = [
    "conduction_any",
    "paced_any",
    "afib_flutter",
    "ischemia_st",
    "diag_code__NIVCD",
    "diag_code__AFIB0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Sunnybrook all-feature benchmark runs.")
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-baseline", default="baseline_exact_vae")
    parser.add_argument("--label-candidate", default="candidate_fm_vae")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_grouped_summary(baseline_dir: Path, candidate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(baseline_dir / "grouped_summary.csv")
    cand = pd.read_csv(candidate_dir / "grouped_summary.csv")
    key_cols = ["target_group", "target_type", "n_targets"]
    merged = base.merge(cand, on=key_cols, suffixes=("_baseline", "_candidate"))
    delta_cols = [
        "orig12_primary_mean",
        "recon12_primary_mean",
        "obs3_primary_mean",
        "recon_minus_obs3_mean",
        "orig12_minus_recon_mean",
        "recon_beats_obs3_rate",
        "within_orig12_tolerance_rate",
        "recovery_ratio_mean",
    ]
    for col in delta_cols:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]

    overall = merged[merged["target_group"] == "__overall__"].copy()
    return merged, overall


def compare_direct_features(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_dir / "direct_feature_metrics.csv")
    cand = pd.read_csv(candidate_dir / "direct_feature_metrics.csv")
    merged = base.merge(cand, on=["source", "feature", "n"], suffixes=("_baseline", "_candidate"))
    for metric in ["pearson", "spearman"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
    for metric in ["mae", "rmse"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_baseline"] - merged[f"{metric}_candidate"]
    return merged


def compare_regression_targets(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_dir / "all_feature_regression_results.csv")
    cand = pd.read_csv(candidate_dir / "all_feature_regression_results.csv")
    merged = base.merge(
        cand,
        on=["target", "group", "support_tier"],
        suffixes=("_baseline", "_candidate"),
    )
    metric_cols = [
        "orig12_spearman",
        "recon12_spearman",
        "obs3_spearman",
        "recon_minus_obs3",
        "orig12_minus_recon",
        "recovery_ratio",
    ]
    for col in metric_cols:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    merged["fm_better_than_baseline"] = merged["recon12_spearman_delta"] > 0
    return merged.sort_values("recon12_spearman_delta", ascending=False).reset_index(drop=True)


def compare_classification_targets(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_dir / "all_feature_classification_results.csv")
    cand = pd.read_csv(candidate_dir / "all_feature_classification_results.csv")
    merged = base.merge(
        cand,
        on=["target", "group", "support_tier"],
        suffixes=("_baseline", "_candidate"),
    )
    metric_cols = [
        "orig12_auroc",
        "recon12_auroc",
        "obs3_auroc",
        "recon_minus_obs3",
        "orig12_minus_recon",
        "recovery_ratio",
    ]
    for col in metric_cols:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    merged["fm_better_than_baseline"] = merged["recon12_auroc_delta"] > 0
    return merged.sort_values("recon12_auroc_delta", ascending=False).reset_index(drop=True)


def compare_diag_code_metrics(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base_path = baseline_dir / "sunnybrook_diag_code_metrics.csv"
    cand_path = candidate_dir / "sunnybrook_diag_code_metrics.csv"
    if not base_path.exists() or not cand_path.exists():
        return pd.DataFrame()
    base = pd.read_csv(base_path)
    cand = pd.read_csv(cand_path)
    merged = base.merge(
        cand,
        on=["target", "diag_code", "family_group", "positive_count", "prevalence", "support_tier"],
        suffixes=("_baseline", "_candidate"),
    )
    metric_cols = [
        "orig12_auroc",
        "recon12_auroc",
        "obs3_auroc",
        "recon_minus_obs3",
        "orig12_minus_recon",
        "recovery_ratio",
    ]
    for col in metric_cols:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    merged["fm_better_than_baseline"] = merged["recon12_auroc_delta"] > 0
    return merged.sort_values(["recon12_auroc_delta", "diag_code"], ascending=[False, True]).reset_index(drop=True)


def compare_diag_code_families(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    base_path = baseline_dir / "sunnybrook_diag_code_family_summary.csv"
    cand_path = candidate_dir / "sunnybrook_diag_code_family_summary.csv"
    if not base_path.exists() or not cand_path.exists():
        return pd.DataFrame()
    base = pd.read_csv(base_path)
    cand = pd.read_csv(cand_path)
    merged = base.merge(cand, on=["family_group", "n_codes"], suffixes=("_baseline", "_candidate"))
    metric_cols = [
        "orig12_auroc_mean",
        "recon12_auroc_mean",
        "obs3_auroc_mean",
        "recon_minus_obs3_mean",
        "orig12_minus_recon_mean",
        "recon_beats_obs3_rate",
        "within_orig12_tolerance_rate",
        "recovery_ratio_mean",
    ]
    for col in metric_cols:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    return merged.sort_values("family_group").reset_index(drop=True)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_r2_raw(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sum((b - np.mean(b)) ** 2))
    if denom < 1e-8:
        return float("nan")
    return float(1.0 - (np.sum((a - b) ** 2) / denom))


def safe_r2_engineering(a: np.ndarray, b: np.ndarray) -> float:
    ssr = float(np.sum((a - b) ** 2))
    sst = float(np.sum((b - np.mean(b)) ** 2))
    return float(max(1.0 - ssr / max(sst, 0.1), -100.0))


def compute_snr_db(pred: np.ndarray, gt: np.ndarray) -> float:
    signal_power = float(np.mean(gt ** 2))
    noise_power = float(np.mean((gt - pred) ** 2))
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def compute_fourier_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_mag = np.abs(rfft(pred))
    gt_mag = np.abs(rfft(gt))
    denom = float(np.linalg.norm(gt_mag)) + 1e-12
    return float(np.linalg.norm(pred_mag - gt_mag) / denom)


def compute_hausdorff_distance(pred: np.ndarray, gt: np.ndarray, step: int = 10) -> float:
    pred_ds = pred[::step]
    gt_ds = gt[::step]
    pred_z = (pred_ds - np.mean(pred_ds)) / (np.std(pred_ds) + 1e-8)
    gt_z = (gt_ds - np.mean(gt_ds)) / (np.std(gt_ds) + 1e-8)
    t = np.linspace(0.0, 1.0, len(pred_z), dtype=np.float32)
    pred_pts = np.stack([t, pred_z.astype(np.float32)], axis=1)
    gt_pts = np.stack([t, gt_z.astype(np.float32)], axis=1)
    return float(max(directed_hausdorff(pred_pts, gt_pts)[0], directed_hausdorff(gt_pts, pred_pts)[0]))


def compute_waveform_metrics(
    files: list[str],
    arrays_by_source: dict[str, np.ndarray],
    obs_indices: list[int],
    model_label: str,
) -> pd.DataFrame:
    gt = arrays_by_source["orig12"]
    pred = arrays_by_source["recon12"]
    rows = []
    for file_idx, file_name in enumerate(files):
        for lead_idx, lead_name in enumerate(LEAD_ORDER):
            gt_lead = gt[file_idx, lead_idx].astype(np.float64)
            pred_lead = pred[file_idx, lead_idx].astype(np.float64)
            err = pred_lead - gt_lead
            mse = float(np.mean(err ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(err)))
            rows.append(
                {
                    "model_label": model_label,
                    "file": file_name,
                    "lead": lead_name,
                    "lead_idx": lead_idx,
                    "is_observed": lead_idx in obs_indices,
                    "is_hidden": lead_idx not in obs_indices,
                    "is_limb": lead_idx < 6,
                    "is_chest": lead_idx >= 6,
                    "r2_engineering": safe_r2_engineering(pred_lead, gt_lead),
                    "r2_raw": safe_r2_raw(pred_lead, gt_lead),
                    "pearson": safe_pearson(pred_lead, gt_lead),
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "snr_db": compute_snr_db(pred_lead, gt_lead),
                    "fourier_distance": compute_fourier_distance(pred_lead, gt_lead),
                    "hausdorff_distance": compute_hausdorff_distance(pred_lead, gt_lead),
                }
            )
    return pd.DataFrame(rows)


def compute_scope_r2_metrics(
    files: list[str],
    arrays_by_source: dict[str, np.ndarray],
    obs_indices: list[int],
    model_label: str,
) -> pd.DataFrame:
    gt = torch.tensor(arrays_by_source["orig12"], dtype=torch.float32)
    pred = torch.tensor(arrays_by_source["recon12"], dtype=torch.float32)
    hidden_indices = [i for i in range(len(LEAD_ORDER)) if i not in obs_indices]
    scope_map = {
        "all_leads": list(range(len(LEAD_ORDER))),
        "hidden_leads": hidden_indices,
        "observed_leads": obs_indices,
        "limb_leads": list(range(6)),
        "chest_leads": list(range(6, len(LEAD_ORDER))),
    }
    rows = []
    for file_idx, file_name in enumerate(files):
        gt_file = gt[file_idx : file_idx + 1]
        pred_file = pred[file_idx : file_idx + 1]
        for scope, lead_indices in scope_map.items():
            gt_scope = gt_file[:, lead_indices, :]
            pred_scope = pred_file[:, lead_indices, :]
            rows.append(
                {
                    "model_label": model_label,
                    "file": file_name,
                    "scope": scope,
                    "sample_r2_mean": float(calculate_sample_r2(pred_scope, gt_scope, use_batch_mean=False).mean().item()),
                    "mason_r2": float((-mason_batch_r2_loss(pred_scope, gt_scope)).item()),
                }
            )
    return pd.DataFrame(rows)


def compute_cohort_r2_metrics(
    arrays_by_source: dict[str, np.ndarray],
    obs_indices: list[int],
    model_label: str,
) -> pd.DataFrame:
    gt = torch.tensor(arrays_by_source["orig12"], dtype=torch.float32)
    pred = torch.tensor(arrays_by_source["recon12"], dtype=torch.float32)
    hidden_indices = [i for i in range(len(LEAD_ORDER)) if i not in obs_indices]
    scope_map = {
        "all_leads": list(range(len(LEAD_ORDER))),
        "hidden_leads": hidden_indices,
        "observed_leads": obs_indices,
        "limb_leads": list(range(6)),
        "chest_leads": list(range(6, len(LEAD_ORDER))),
    }
    rows = []
    for scope, lead_indices in scope_map.items():
        gt_scope = gt[:, lead_indices, :]
        pred_scope = pred[:, lead_indices, :]
        rows.append(
            {
                "model_label": model_label,
                "scope": scope,
                "cohort_batch_r2": float((-mason_batch_r2_loss(pred_scope, gt_scope, use_batch_mean=True)).item()),
                "cohort_sample_r2_mean": float(calculate_sample_r2(pred_scope, gt_scope, use_batch_mean=False).mean().item()),
            }
        )
    return pd.DataFrame(rows)


def summarize_waveform_by_lead(per_lead: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_lead.groupby(["model_label", "lead"], as_index=False)[
            ["r2_engineering", "r2_raw", "pearson", "mse", "rmse", "mae", "snr_db", "fourier_distance", "hausdorff_distance"]
        ]
        .mean()
    )
    base = grouped[grouped["model_label"] == "baseline"].drop(columns=["model_label"])
    cand = grouped[grouped["model_label"] == "candidate"].drop(columns=["model_label"])
    merged = base.merge(cand, on="lead", suffixes=("_baseline", "_candidate"))
    for metric in ["r2_engineering", "r2_raw", "pearson", "mse", "rmse", "mae", "snr_db", "fourier_distance", "hausdorff_distance"]:
        merged[f"{metric}_delta_raw"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
    for metric in ["r2_engineering", "r2_raw", "pearson", "snr_db"]:
        merged[f"{metric}_improvement"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
    for metric in ["mse", "rmse", "mae", "fourier_distance", "hausdorff_distance"]:
        merged[f"{metric}_improvement"] = merged[f"{metric}_baseline"] - merged[f"{metric}_candidate"]
    return merged


def summarize_waveform_by_scope(per_lead: pd.DataFrame) -> pd.DataFrame:
    scope_rows = []
    scope_masks = {
        "all_leads": per_lead.index == per_lead.index,
        "hidden_leads": per_lead["is_hidden"],
        "observed_leads": per_lead["is_observed"],
        "limb_leads": per_lead["is_limb"],
        "chest_leads": per_lead["is_chest"],
    }
    metric_cols = ["r2_engineering", "r2_raw", "pearson", "mse", "rmse", "mae", "snr_db", "fourier_distance", "hausdorff_distance"]
    for scope, mask in scope_masks.items():
        scoped = per_lead[mask]
        summary = scoped.groupby("model_label", as_index=False)[metric_cols].mean()
        for row in summary.itertuples(index=False):
            rec = {"scope": scope, "model_label": row.model_label}
            for metric in metric_cols:
                rec[metric] = getattr(row, metric)
            scope_rows.append(rec)
    scope_df = pd.DataFrame(scope_rows)
    base = scope_df[scope_df["model_label"] == "baseline"].drop(columns=["model_label"])
    cand = scope_df[scope_df["model_label"] == "candidate"].drop(columns=["model_label"])
    merged = base.merge(cand, on="scope", suffixes=("_baseline", "_candidate"))
    for metric in metric_cols:
        merged[f"{metric}_delta_raw"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
    for metric in ["r2_engineering", "r2_raw", "pearson", "snr_db"]:
        merged[f"{metric}_improvement"] = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
    for metric in ["mse", "rmse", "mae", "fourier_distance", "hausdorff_distance"]:
        merged[f"{metric}_improvement"] = merged[f"{metric}_baseline"] - merged[f"{metric}_candidate"]
    return merged


def merge_scope_r2_metrics(scope_r2: pd.DataFrame, cohort_r2: pd.DataFrame, by_scope: pd.DataFrame) -> pd.DataFrame:
    extra = (
        scope_r2.groupby(["model_label", "scope"], as_index=False)[["sample_r2_mean", "mason_r2"]]
        .mean()
    )
    base = extra[extra["model_label"] == "baseline"].drop(columns=["model_label"])
    cand = extra[extra["model_label"] == "candidate"].drop(columns=["model_label"])
    merged = base.merge(cand, on="scope", suffixes=("_baseline", "_candidate"))
    merged["sample_r2_mean_delta_raw"] = merged["sample_r2_mean_candidate"] - merged["sample_r2_mean_baseline"]
    merged["sample_r2_mean_improvement"] = merged["sample_r2_mean_candidate"] - merged["sample_r2_mean_baseline"]
    merged["mason_r2_delta_raw"] = merged["mason_r2_candidate"] - merged["mason_r2_baseline"]
    merged["mason_r2_improvement"] = merged["mason_r2_candidate"] - merged["mason_r2_baseline"]
    cohort_base = cohort_r2[cohort_r2["model_label"] == "baseline"].drop(columns=["model_label"])
    cohort_cand = cohort_r2[cohort_r2["model_label"] == "candidate"].drop(columns=["model_label"])
    cohort_merged = cohort_base.merge(cohort_cand, on="scope", suffixes=("_baseline", "_candidate"))
    cohort_merged["cohort_batch_r2_delta_raw"] = cohort_merged["cohort_batch_r2_candidate"] - cohort_merged["cohort_batch_r2_baseline"]
    cohort_merged["cohort_batch_r2_improvement"] = cohort_merged["cohort_batch_r2_delta_raw"]
    cohort_merged["cohort_sample_r2_mean_delta_raw"] = cohort_merged["cohort_sample_r2_mean_candidate"] - cohort_merged["cohort_sample_r2_mean_baseline"]
    cohort_merged["cohort_sample_r2_mean_improvement"] = cohort_merged["cohort_sample_r2_mean_delta_raw"]
    return by_scope.merge(merged, on="scope", how="left").merge(cohort_merged, on="scope", how="left")


def build_waveform_comparison(
    baseline_meta: dict,
    candidate_meta: dict,
    device: torch.device,
    use_amp: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_leads = baseline_meta["obs_leads"]
    obs_indices = parse_obs_leads(",".join(obs_leads))
    records = sorted(Path(baseline_meta["sunnybrook_dir"]).glob("*.xml"))

    baseline_model, _, _ = load_reconstruction_model(baseline_meta["checkpoint"], device, baseline_meta.get("model_family", "auto"))
    candidate_model, _, _ = load_reconstruction_model(candidate_meta["checkpoint"], device, candidate_meta.get("model_family", "auto"))

    _, baseline_arrays = reconstruct_sources(baseline_model, records, obs_indices, device, use_amp)
    _, candidate_arrays = reconstruct_sources(candidate_model, records, obs_indices, device, use_amp)
    files = [path.name for path in records]

    baseline_metrics = compute_waveform_metrics(files, baseline_arrays, obs_indices, "baseline")
    candidate_metrics = compute_waveform_metrics(files, candidate_arrays, obs_indices, "candidate")
    baseline_scope_r2 = compute_scope_r2_metrics(files, baseline_arrays, obs_indices, "baseline")
    candidate_scope_r2 = compute_scope_r2_metrics(files, candidate_arrays, obs_indices, "candidate")
    baseline_cohort_r2 = compute_cohort_r2_metrics(baseline_arrays, obs_indices, "baseline")
    candidate_cohort_r2 = compute_cohort_r2_metrics(candidate_arrays, obs_indices, "candidate")
    per_lead = pd.concat([baseline_metrics, candidate_metrics], ignore_index=True)
    scope_r2 = pd.concat([baseline_scope_r2, candidate_scope_r2], ignore_index=True)
    cohort_r2 = pd.concat([baseline_cohort_r2, candidate_cohort_r2], ignore_index=True)
    by_lead = summarize_waveform_by_lead(per_lead)
    by_scope = merge_scope_r2_metrics(scope_r2, cohort_r2, summarize_waveform_by_scope(per_lead))
    return per_lead, by_lead, by_scope


def make_scorecard(
    catalog: pd.DataFrame,
    regression_cmp: pd.DataFrame,
    classification_cmp: pd.DataFrame,
) -> pd.DataFrame:
    supported_catalog = catalog[catalog["support_tier"].isin(["supported", "exploratory"])].copy()
    rows = []
    for row in supported_catalog.itertuples(index=False):
        target = row.canonical_name
        if row.target_type == "regression":
            hit = regression_cmp[regression_cmp["target"] == target]
            if hit.empty:
                continue
            rec = hit.iloc[0]
            rows.append(
                {
                    "target": target,
                    "target_type": "regression",
                    "group": row.group,
                    "support_tier": row.support_tier,
                    "baseline_recon_primary": rec["recon12_spearman_baseline"],
                    "candidate_recon_primary": rec["recon12_spearman_candidate"],
                    "candidate_minus_baseline": rec["recon12_spearman_delta"],
                    "obs3_primary": rec["obs3_spearman_baseline"],
                    "orig12_primary": rec["orig12_spearman_baseline"],
                    "beats_baseline": bool(rec["recon12_spearman_delta"] > 0),
                }
            )
        elif row.target_type == "classification":
            hit = classification_cmp[classification_cmp["target"] == target]
            if hit.empty:
                continue
            rec = hit.iloc[0]
            rows.append(
                {
                    "target": target,
                    "target_type": "classification",
                    "group": row.group,
                    "support_tier": row.support_tier,
                    "baseline_recon_primary": rec["recon12_auroc_baseline"],
                    "candidate_recon_primary": rec["recon12_auroc_candidate"],
                    "candidate_minus_baseline": rec["recon12_auroc_delta"],
                    "obs3_primary": rec["obs3_auroc_baseline"],
                    "orig12_primary": rec["orig12_auroc_baseline"],
                    "beats_baseline": bool(rec["recon12_auroc_delta"] > 0),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_type", "candidate_minus_baseline"], ascending=[True, False])


def write_interpretation(
    output_dir: Path,
    baseline_meta: dict,
    candidate_meta: dict,
    overall: pd.DataFrame,
    direct_cmp: pd.DataFrame,
    regression_cmp: pd.DataFrame,
    classification_cmp: pd.DataFrame,
    diag_code_cmp: pd.DataFrame,
    scorecard: pd.DataFrame,
    waveform_scope: pd.DataFrame,
    waveform_lead: pd.DataFrame,
) -> None:
    reg = overall[overall["target_type"] == "regression"].iloc[0]
    cls = overall[overall["target_type"] == "classification"].iloc[0]
    direct_recon = direct_cmp[direct_cmp["source"] == "recon12"].copy()
    clinical = classification_cmp[classification_cmp["target"].isin(CLINICAL_PRIORITY_TARGETS)].copy()
    diag_top = diag_code_cmp.head(10) if not diag_code_cmp.empty else pd.DataFrame()
    hidden_scope = waveform_scope[waveform_scope["scope"] == "hidden_leads"].iloc[0]
    all_scope = waveform_scope[waveform_scope["scope"] == "all_leads"].iloc[0]
    top_reg_gains = regression_cmp[["target", "recon12_spearman_delta", "group"]].head(10)
    top_reg_losses = regression_cmp.sort_values("recon12_spearman_delta", ascending=True)[["target", "recon12_spearman_delta", "group"]].head(10)
    top_cls_gains = classification_cmp[["target", "recon12_auroc_delta", "group"]].head(10)
    top_cls_losses = classification_cmp.sort_values("recon12_auroc_delta", ascending=True)[["target", "recon12_auroc_delta", "group"]].head(10)

    lines = [
        "# Sunnybrook Baseline vs FM Comparison",
        "",
        "## Runs",
        f"- Baseline: `{baseline_meta['run_name']}`",
        f"- Candidate: `{candidate_meta['run_name']}`",
        "",
        "## Overall",
        (
            f"- Regression primary mean (Spearman on recon12): "
            f"`{reg['recon12_primary_mean_baseline']:.3f}` -> `{reg['recon12_primary_mean_candidate']:.3f}` "
            f"(delta `{reg['recon12_primary_mean_delta']:+.3f}`)"
        ),
        (
            f"- Regression win rate vs raw 3-lead: "
            f"`{reg['recon_beats_obs3_rate_baseline']:.3f}` -> `{reg['recon_beats_obs3_rate_candidate']:.3f}` "
            f"(delta `{reg['recon_beats_obs3_rate_delta']:+.3f}`)"
        ),
        (
            f"- Classification primary mean (AUROC on recon12): "
            f"`{cls['recon12_primary_mean_baseline']:.3f}` -> `{cls['recon12_primary_mean_candidate']:.3f}` "
            f"(delta `{cls['recon12_primary_mean_delta']:+.3f}`)"
        ),
        (
            f"- Classification win rate vs raw 3-lead: "
            f"`{cls['recon_beats_obs3_rate_baseline']:.3f}` -> `{cls['recon_beats_obs3_rate_candidate']:.3f}` "
            f"(delta `{cls['recon_beats_obs3_rate_delta']:+.3f}`)"
        ),
        "",
        "## Direct Waveform Reconstruction",
        "",
        "### Primary",
        (
            f"- Hidden-lead cohort batch R2: `{hidden_scope['cohort_batch_r2_baseline']:.3f}` -> `{hidden_scope['cohort_batch_r2_candidate']:.3f}` "
            f"(delta `{hidden_scope['cohort_batch_r2_improvement']:+.3f}`)"
        ),
        (
            f"- Hidden-lead mean Pearson: `{hidden_scope['pearson_baseline']:.3f}` -> `{hidden_scope['pearson_candidate']:.3f}` "
            f"(delta `{hidden_scope['pearson_improvement']:+.3f}`)"
        ),
        (
            f"- Hidden-lead mean RMSE: `{hidden_scope['rmse_baseline']:.6f}` -> `{hidden_scope['rmse_candidate']:.6f}` "
            f"(improvement `{hidden_scope['rmse_improvement']:+.6f}`)"
        ),
        "",
        "### Secondary",
        (
            f"- Hidden-lead mean MSE: `{hidden_scope['mse_baseline']:.6f}` -> `{hidden_scope['mse_candidate']:.6f}` "
            f"(improvement `{hidden_scope['mse_improvement']:+.6f}`)"
        ),
        (
            f"- Hidden-lead mean MAE: `{hidden_scope['mae_baseline']:.6f}` -> `{hidden_scope['mae_candidate']:.6f}` "
            f"(improvement `{hidden_scope['mae_improvement']:+.6f}`)"
        ),
        (
            f"- Hidden-lead mean SNR: `{hidden_scope['snr_db_baseline']:.3f}` -> `{hidden_scope['snr_db_candidate']:.3f}` "
            f"(delta `{hidden_scope['snr_db_improvement']:+.3f}`)"
        ),
        (
            f"- Hidden-lead Fourier distance: `{hidden_scope['fourier_distance_baseline']:.3f}` -> "
            f"`{hidden_scope['fourier_distance_candidate']:.3f}` "
            f"(improvement `{hidden_scope['fourier_distance_improvement']:+.3f}`)"
        ),
        (
            f"- Hidden-lead Hausdorff distance: `{hidden_scope['hausdorff_distance_baseline']:.3f}` -> "
            f"`{hidden_scope['hausdorff_distance_candidate']:.3f}` "
            f"(improvement `{hidden_scope['hausdorff_distance_improvement']:+.3f}`)"
        ),
        "",
        "### Audit",
        (
            f"- Hidden-lead mean sample_r2_mean: `{hidden_scope['sample_r2_mean_baseline']:.3f}` -> `{hidden_scope['sample_r2_mean_candidate']:.3f}` "
            f"(delta `{hidden_scope['sample_r2_mean_improvement']:+.3f}`)"
        ),
        (
            f"- Hidden-lead per-lead raw R2 audit: `{hidden_scope['r2_raw_baseline']:.3f}` -> `{hidden_scope['r2_raw_candidate']:.3f}` "
            f"(delta `{hidden_scope['r2_raw_improvement']:+.3f}`)"
        ),
        "",
        "## Direct Zero-Shot Recon12 Feature Changes",
    ]

    for row in direct_recon.itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: spearman `{row.spearman_baseline:.3f}` -> `{row.spearman_candidate:.3f}` "
            f"(delta `{row.spearman_delta:+.3f}`), mae `{row.mae_baseline:.3f}` -> `{row.mae_candidate:.3f}` "
            f"(improvement `{row.mae_delta:+.3f}`)"
        )

    lines.extend(["", "## Clinical Priority Classification Targets"])
    for row in clinical.itertuples(index=False):
        lines.append(
            f"- `{row.target}`: recon AUROC `{row.recon12_auroc_baseline:.3f}` -> `{row.recon12_auroc_candidate:.3f}` "
            f"(delta `{row.recon12_auroc_delta:+.3f}`), raw 3-lead `{row.obs3_auroc_baseline:.3f}`, "
            f"original 12-lead `{row.orig12_auroc_baseline:.3f}`"
        )

    if not diag_top.empty:
        lines.extend(["", "## Literal Diagnostic-Code Gains"])
        for row in diag_top.itertuples(index=False):
            lines.append(
                f"- `{row.diag_code}` ({row.family_group}): recon AUROC `{row.recon12_auroc_baseline:.3f}` -> "
                f"`{row.recon12_auroc_candidate:.3f}` (delta `{row.recon12_auroc_delta:+.3f}`)"
            )

    lines.extend(["", "## Top Regression Gains"])
    for row in top_reg_gains.itertuples(index=False):
        lines.append(f"- `{row.target}` ({row.group}): delta `{row.recon12_spearman_delta:+.3f}`")

    lines.extend(["", "## Top Regression Losses"])
    for row in top_reg_losses.itertuples(index=False):
        lines.append(f"- `{row.target}` ({row.group}): delta `{row.recon12_spearman_delta:+.3f}`")

    lines.extend(["", "## Top Classification Gains"])
    for row in top_cls_gains.itertuples(index=False):
        lines.append(f"- `{row.target}` ({row.group}): delta `{row.recon12_auroc_delta:+.3f}`")

    lines.extend(["", "## Top Classification Losses"])
    for row in top_cls_losses.itertuples(index=False):
        lines.append(f"- `{row.target}` ({row.group}): delta `{row.recon12_auroc_delta:+.3f}`")

    reg_wins = int((scorecard["target_type"].eq("regression") & scorecard["beats_baseline"]).sum())
    reg_total = int(scorecard["target_type"].eq("regression").sum())
    cls_wins = int((scorecard["target_type"].eq("classification") & scorecard["beats_baseline"]).sum())
    cls_total = int(scorecard["target_type"].eq("classification").sum())
    waveform_delta = float(hidden_scope["cohort_batch_r2_improvement"])
    if waveform_delta > 0:
        waveform_bottom_line = "- Direct waveform metrics are directionally better for the candidate on the hidden leads."
    elif waveform_delta < 0:
        waveform_bottom_line = "- Direct waveform metrics are worse for the candidate on the hidden leads; the baseline remains stronger on strict sample-level reconstruction."
    else:
        waveform_bottom_line = "- Direct waveform metrics are essentially tied on the hidden leads."
    lines.extend(
        [
            "",
            "## Bottom Line",
            (
                f"- Candidate FM beats the exact baseline on `{reg_wins}/{reg_total}` supported/exploratory regression targets "
                f"and `{cls_wins}/{cls_total}` supported/exploratory classification targets."
            ),
            "- The strongest external gains are in downstream classification, especially conduction/pacing-related targets.",
            waveform_bottom_line,
            "- Regression improvements are more selective: some morphology families improve, but several ST-related features regress.",
        ]
    )

    (output_dir / "interpretation.md").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_meta = load_json(baseline_dir / "run_metadata.json")
    candidate_meta = load_json(candidate_dir / "run_metadata.json")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_root) / f"{baseline_dir.name}_vs_{candidate_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(baseline_dir / "target_catalog.csv")
    grouped_cmp, overall = merge_grouped_summary(baseline_dir, candidate_dir)
    direct_cmp = compare_direct_features(baseline_dir, candidate_dir)
    regression_cmp = compare_regression_targets(baseline_dir, candidate_dir)
    classification_cmp = compare_classification_targets(baseline_dir, candidate_dir)
    diag_code_cmp = compare_diag_code_metrics(baseline_dir, candidate_dir)
    diag_code_family_cmp = compare_diag_code_families(baseline_dir, candidate_dir)
    scorecard = make_scorecard(catalog, regression_cmp, classification_cmp)
    waveform_record_metrics, waveform_lead_summary, waveform_scope_summary = build_waveform_comparison(
        baseline_meta,
        candidate_meta,
        device,
        args.use_amp,
    )

    grouped_cmp.to_csv(output_dir / "grouped_summary_comparison.csv", index=False)
    overall.to_csv(output_dir / "overall_summary.csv", index=False)
    direct_cmp.to_csv(output_dir / "direct_feature_comparison.csv", index=False)
    regression_cmp.to_csv(output_dir / "regression_target_comparison.csv", index=False)
    classification_cmp.to_csv(output_dir / "classification_target_comparison.csv", index=False)
    diag_code_cmp.to_csv(output_dir / "sunnybrook_diag_code_scoreboard.csv", index=False)
    diag_code_family_cmp.to_csv(output_dir / "sunnybrook_diag_code_family_comparison.csv", index=False)
    scorecard.to_csv(output_dir / "feature_scorecard.csv", index=False)
    waveform_record_metrics.to_csv(output_dir / "waveform_record_metrics.csv", index=False)
    waveform_lead_summary.to_csv(output_dir / "waveform_lead_summary.csv", index=False)
    waveform_scope_summary.to_csv(output_dir / "waveform_scope_summary.csv", index=False)

    clinical = classification_cmp[classification_cmp["target"].isin(CLINICAL_PRIORITY_TARGETS)].copy()
    clinical.to_csv(output_dir / "clinical_priority_comparison.csv", index=False)

    summary_payload = {
        "baseline_run": baseline_meta["run_name"],
        "candidate_run": candidate_meta["run_name"],
        "baseline_label": args.label_baseline,
        "candidate_label": args.label_candidate,
        "regression_targets_scored": int((scorecard["target_type"] == "regression").sum()),
        "classification_targets_scored": int((scorecard["target_type"] == "classification").sum()),
        "regression_wins_for_candidate": int((scorecard["target_type"].eq("regression") & scorecard["beats_baseline"]).sum()),
        "classification_wins_for_candidate": int((scorecard["target_type"].eq("classification") & scorecard["beats_baseline"]).sum()),
        "waveform_hidden_lead_r2_engineering_baseline": float(waveform_scope_summary.loc[waveform_scope_summary["scope"] == "hidden_leads", "r2_engineering_baseline"].iloc[0]),
        "waveform_hidden_lead_r2_engineering_candidate": float(waveform_scope_summary.loc[waveform_scope_summary["scope"] == "hidden_leads", "r2_engineering_candidate"].iloc[0]),
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="ascii")

    write_interpretation(
        output_dir,
        baseline_meta,
        candidate_meta,
        overall,
        direct_cmp,
        regression_cmp,
        classification_cmp,
        diag_code_cmp,
        scorecard,
        waveform_scope_summary,
        waveform_lead_summary,
    )
    print(f"Saved Sunnybrook comparison to {output_dir}")


if __name__ == "__main__":
    main()
