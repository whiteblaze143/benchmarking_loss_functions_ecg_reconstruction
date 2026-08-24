"""Markdown reporting utilities for ECG reconstruction evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.reconstruction.evaluate_functions.analysis import BlandAltmanResult


def _format_metric(value: float, ci: Optional[tuple[float, float]] = None, unit: str = "") -> str:
    if ci:
        return f"{value:.4f}{unit} ± {(ci[1] - ci[0]) / 2:.4f}"
    return f"{value:.4f}{unit}"


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def generate_report(
    summary_metrics: Dict[str, float],
    per_lead_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    baseline_results: Dict[str, float],
    linear_baseline_metrics: Optional[Dict[str, float]],
    bland_altman: Dict[str, BlandAltmanResult],
    ablation_df: pd.DataFrame,
    clinical_df: Optional[pd.DataFrame],
    save_path: Path,
    additional_notes: Optional[List[str]] = None,
) -> None:
    lines: List[str] = []
    lines.append("# ECG Reconstruction Evaluation Summary\n")
    lines.append("## Aggregate Metrics\n")
    lines.append(f"- MSE: {summary_metrics['mse']:.4f}")
    lines.append(f"- MAE: {summary_metrics['mae']:.4f}")
    lines.append(f"- RMSE: {summary_metrics['rmse']:.4f}")
    lines.append(f"- Pearson r: {summary_metrics['pearson_r']:.4f}\n")

    lines.append("## Per-Lead Performance\n")
    lines.append(_df_to_markdown(per_lead_df))
    lines.append("")

    lines.append("## Baseline Comparison\n")
    lines.append(f"- Conditional ResCNN best val loss: {baseline_results.get('model_val_loss', float('nan')):.4f}")
    if linear_baseline_metrics:
        lines.append(f"- Linear regression MSE: {linear_baseline_metrics.get('mse', float('nan')):.4f}")
    else:
        lines.append("- Linear regression baseline not computed.")
    if "mason" in baseline_results:
        lines.append(f"- Mason baseline MSE: {baseline_results['mason']:.4f}")
    else:
        lines.append("- Mason baseline not available.")
    lines.append("")

    lines.append("## Bland–Altman Highlights\n")
    if bland_altman:
        lines.append("| Lead | Bias (mV) | LoA Lower | LoA Upper | Slope | Corr |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for lead, result in bland_altman.items():
            lines.append(
                f"| {lead} | {result.bias:.4f} | {result.loa_lower:.4f} | {result.loa_upper:.4f} | {result.slope:.4f} | {result.correlation:.4f} |"
            )
        lines.append("")
    else:
        lines.append("_No Bland–Altman statistics computed._\n")

    lines.append("## Subgroup Analysis\n")
    lines.append(_df_to_markdown(subgroup_df))
    lines.append("")

    lines.append("## Feature Importance (Ablation)\n")
    lines.append(_df_to_markdown(ablation_df))
    lines.append("")

    if clinical_df is not None:
        lines.append("## Clinical Cohorts\n")
        lines.append(_df_to_markdown(clinical_df))
        lines.append("")

    if additional_notes:
        lines.append("## Notes\n")
        for note in additional_notes:
            lines.append(f"- {note}")
        lines.append("")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")

