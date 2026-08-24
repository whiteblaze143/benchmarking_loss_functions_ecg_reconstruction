#!/usr/bin/env python3
"""Audit factorial coverage, estimate paired effects, and make poster artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap, norm
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("unet", "msvae", "ecgaim")
FAMILY_LABELS = {"unet": "U-Net", "msvae": "MultiScale-VAE", "ecgaim": "ECG-AIM"}
FAMILY_COLORS = {"unet": "#0072B2", "msvae": "#E69F00", "ecgaim": "#009E73"}
MASKS = [f"{value:04b}" for value in range(16)]
COMPONENTS = {"mse": 0, "correlation": 1, "mmd": 2, "derivative": 3}
PRIMARY_MASKS = [f"1{value:03b}" for value in range(8)]
PRIMARY_COMPONENTS = {"correlation": 1, "mmd": 2, "derivative": 3}
CONFIRMATION_SEEDS = (1337, 2026)
CONFIRMATION_SLOTS = ("base", "full", "best")
ECGFOUNDER_NI_MARGIN = 0.02
ECHONEXT_SHD_LABELS = (
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
)


def strict_write(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bca_mean(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return {"estimate": safe_float(values.mean()) if values.size else None, "ci_low": None, "ci_high": None, "n": int(values.size)}
    # SciPy's BCa acceleration is undefined for a constant sample.  The
    # sampling distribution is degenerate in that case, so its exact interval
    # is the common value rather than NaN (and no warning is warranted).
    if np.all(values == values[0]):
        estimate = float(values[0])
        return {"estimate": estimate, "ci_low": estimate, "ci_high": estimate, "n": int(values.size)}
    result = bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=2000,
        method="BCa",
        random_state=np.random.default_rng(42),
    )
    return {
        "estimate": float(values.mean()),
        "ci_low": float(result.confidence_interval.low),
        "ci_high": float(result.confidence_interval.high),
        "n": int(values.size),
    }


def patient_ids_for_ecgs(ecg_ids: pd.Series | np.ndarray) -> np.ndarray:
    metadata = pd.read_csv(
        PROJECT_ROOT / "data/ptb_xl/ptbxl_database.csv",
        usecols=["ecg_id", "patient_id"],
    ).set_index("ecg_id")
    mapped = pd.Series(np.asarray(ecg_ids, dtype=np.int64)).map(
        metadata["patient_id"]
    )
    if mapped.isna().any():
        raise ValueError("Every PTB-XL ECG must map to a patient_id")
    return mapped.to_numpy(dtype=np.int64)


def cluster_bca_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str | None]:
    """BCa mean interval with patients, rather than ECG rows, resampled."""
    frame = pd.DataFrame({
        "value": np.asarray(values, dtype=np.float64),
        "cluster": np.asarray(clusters),
    }).dropna(subset=["value", "cluster"])
    if frame.empty:
        return {
            "estimate": None, "ci_low": None, "ci_high": None,
            "n": 0, "n_patients": 0,
            "method": "paired_patient_cluster_BCa",
        }
    grouped = frame.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    n_clusters = len(grouped)
    estimate = float(sums.sum() / counts.sum())
    if n_clusters < 3 or np.all(frame["value"].to_numpy() == estimate):
        return {
            "estimate": estimate,
            "ci_low": estimate if n_clusters else None,
            "ci_high": estimate if n_clusters else None,
            "n": int(len(frame)),
            "n_patients": int(n_clusters),
            "n_resamples": int(n_resamples),
            "method": "paired_patient_cluster_BCa",
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, n_clusters, size=(n_resamples, n_clusters), dtype=np.int32
    )
    distribution = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    less = (
        np.count_nonzero(distribution < estimate)
        + 0.5 * np.count_nonzero(distribution == estimate)
    ) / n_resamples
    z0 = float(norm.ppf(np.clip(less, 1e-8, 1 - 1e-8)))
    jackknife = (sums.sum() - sums) / (counts.sum() - counts)
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.square(centered).sum() ** 1.5)
    acceleration = (
        float(np.power(centered, 3).sum() / denominator)
        if denominator else 0.0
    )
    adjusted = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        divisor = 1.0 - acceleration * (z0 + z_alpha)
        adjusted.append(float(norm.cdf(z0 + (z0 + z_alpha) / divisor)))
    low, high = np.quantile(distribution, np.clip(adjusted, 0.0, 1.0))
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(frame)),
        "n_patients": int(n_clusters),
        "n_resamples": int(n_resamples),
        "method": "paired_patient_cluster_BCa",
        "acceleration": acceleration,
        "bias_correction": z0,
    }


def _auc_and_influence(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, np.ndarray]:
    """Return macro AUROC and its record-level empirical influence function."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    n_records, n_tasks = labels.shape
    valid_aurocs: list[float] = []
    task_influences: list[np.ndarray] = []
    for task in range(n_tasks):
        truth = labels[:, task].astype(bool)
        n_positive = int(truth.sum())
        n_negative = n_records - n_positive
        if n_positive == 0 or n_negative == 0:
            continue
        scores = probabilities[:, task].astype(np.float64)
        auc = float(roc_auc_score(truth, scores))
        negatives = np.sort(scores[~truth])
        positives = np.sort(scores[truth])
        influence = np.empty(n_records, dtype=np.float64)
        positive_scores = scores[truth]
        negative_scores = scores[~truth]
        positive_contribution = (
            np.searchsorted(negatives, positive_scores, side="left")
            + 0.5 * (
                np.searchsorted(negatives, positive_scores, side="right")
                - np.searchsorted(negatives, positive_scores, side="left")
            )
        ) / n_negative
        negative_contribution = (
            n_positive
            - np.searchsorted(positives, negative_scores, side="right")
            + 0.5 * (
                np.searchsorted(positives, negative_scores, side="right")
                - np.searchsorted(positives, negative_scores, side="left")
            )
        ) / n_positive
        prevalence = n_positive / n_records
        influence[truth] = (positive_contribution - auc) / prevalence
        influence[~truth] = (negative_contribution - auc) / (1.0 - prevalence)
        influence -= influence.mean()
        valid_aurocs.append(auc)
        task_influences.append(influence)
    if not valid_aurocs:
        raise ValueError("No ECGFounder task has both positive and negative labels")
    return float(np.mean(valid_aurocs)), np.mean(task_influences, axis=0)


def _paired_influence_bca(
    estimate: float,
    paired_influence: np.ndarray,
    patient_ids: np.ndarray | None = None,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """BCa interval from paired record resamples of the AUROC influence curve."""
    values = np.asarray(paired_influence, dtype=np.float64)
    if patient_ids is None:
        patient_ids = np.arange(values.size, dtype=np.int64)
    interval = cluster_bca_mean(
        values,
        patient_ids,
        n_resamples=n_resamples,
        seed=seed,
    )
    # The influence curve has mean zero; shift its cluster-bootstrap interval
    # onto the observed AUROC difference.
    center = float(interval["estimate"])
    interval["ci_low"] = float(estimate + float(interval["ci_low"]) - center)
    interval["ci_high"] = float(estimate + float(interval["ci_high"]) - center)
    interval["estimate"] = float(estimate)
    interval["n_records"] = interval.pop("n")
    interval["method"] = "paired_patient_cluster_influence_function_BCa"
    return interval


def ecgfounder_noninferiority(
    results: dict[str, Any], registry: dict[str, Any]
) -> list[dict[str, Any]]:
    specs = {spec["id"]: spec for spec in registry["models"]}
    rows: list[dict[str, Any]] = []
    for model_id in sorted(specs):
        model = results["models"][model_id]
        path = PROJECT_ROOT / model["ecgfounder_per_record_parquet"]
        frame = pd.read_parquet(path)
        labels = np.stack(frame["labels"].to_numpy())
        reconstructed = np.stack(frame["probabilities"].to_numpy())
        reference = np.stack(frame["reference_probabilities"].to_numpy())
        reconstructed_auc, reconstructed_if = _auc_and_influence(labels, reconstructed)
        reference_auc, reference_if = _auc_and_influence(labels, reference)
        interval = _paired_influence_bca(
            reconstructed_auc - reference_auc,
            reconstructed_if - reference_if,
            patient_ids_for_ecgs(frame["ecg_id"]),
        )
        rows.append({
            "id": model_id,
            "family": model_id.split("__", 1)[0],
            "mask": specs[model_id]["factorial_mask"],
            "reconstructed_macro_auroc": reconstructed_auc,
            "reference_macro_auroc": reference_auc,
            "difference": interval["estimate"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "margin": ECGFOUNDER_NI_MARGIN,
            "noninferior": bool(interval["ci_low"] > -ECGFOUNDER_NI_MARGIN),
            "n_records": interval["n_records"],
            "n_patients": interval["n_patients"],
            "n_resamples": interval["n_resamples"],
            "method": interval["method"],
        })
    return rows


def confirmation_table(confirmation_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for slot in CONFIRMATION_SLOTS:
            for seed in CONFIRMATION_SEEDS:
                marker = confirmation_dir / f"{family}__{slot}__s{seed}.json"
                if not marker.exists():
                    continue
                payload = json.loads(marker.read_text())
                metrics = payload.get("validation_metrics", {})
                rows.append({
                    "family": family,
                    "slot": slot,
                    "mask": payload.get("mask"),
                    "seed": seed,
                    "checkpoint": payload.get("checkpoint"),
                    "checkpoint_sha256": payload.get("sha256"),
                    "n_samples": metrics.get("n_samples"),
                    "validation_mse": metrics.get("mse"),
                    "validation_r2": metrics.get("r2"),
                    "validation_pearson": metrics.get("pearson"),
                })
    return pd.DataFrame(rows)


def contrast_vector(
    frame: pd.DataFrame,
    component_indices: tuple[int, ...],
    metric: str,
    masks: list[str] = MASKS,
) -> np.ndarray:
    pivot = frame.pivot(index="ecg_id", columns="mask", values=metric)
    required = set(masks)
    if not required.issubset(pivot.columns):
        return np.asarray([], dtype=np.float64)
    signs = {}
    for mask in masks:
        sign = 1
        for index in component_indices:
            sign *= 1 if mask[index] == "1" else -1
        signs[mask] = sign
    factor_count = int(round(math.log2(len(masks))))
    scale = 2 ** (factor_count - len(component_indices))
    return sum(signs[mask] * pivot[mask].to_numpy() for mask in masks) / scale


def contrast_frame(
    frame: pd.DataFrame,
    component_indices: tuple[int, ...],
    metric: str,
    masks: list[str] = MASKS,
) -> pd.DataFrame:
    pivot = frame.pivot(index="ecg_id", columns="mask", values=metric)
    if not set(masks).issubset(pivot.columns):
        return pd.DataFrame(columns=["ecg_id", "patient_id", "value"])
    values = contrast_vector(frame, component_indices, metric, masks=masks)
    return pd.DataFrame({
        "ecg_id": pivot.index.to_numpy(dtype=np.int64),
        "patient_id": patient_ids_for_ecgs(pivot.index.to_numpy()),
        "value": values,
    })


def load_record_table(results: dict[str, Any], registry: dict[str, Any]) -> pd.DataFrame:
    rows = []
    specs = {spec["id"]: spec for spec in registry["models"]}
    for model_id, result in results["models"].items():
        spec = specs[model_id]
        path = PROJECT_ROOT / result["per_record_parquet"]
        frame = pd.read_parquet(path)
        frame = frame[frame["condition"] == "clean"].copy()
        frame["model_id"] = model_id
        frame["family"] = model_id.split("__", 1)[0]
        frame["mask"] = spec["factorial_mask"]
        clinical_path = PROJECT_ROOT / result["ecgfounder_per_record_parquet"]
        clinical = pd.read_parquet(clinical_path)
        clinical_labels = np.stack(clinical["labels"].to_numpy())
        clinical_probabilities = np.stack(clinical["probabilities"].to_numpy())
        clinical_auc, clinical_influence = _auc_and_influence(
            clinical_labels, clinical_probabilities
        )
        diagnostic = pd.DataFrame({
            "ecg_id": clinical["ecg_id"].astype(np.int64),
            "ecgfounder_auroc": clinical_auc + clinical_influence,
        })
        frame = frame.merge(diagnostic, on="ecg_id", how="left", validate="many_to_one")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def factorial_effects(
    records: pd.DataFrame,
    *,
    components: dict[str, int] = PRIMARY_COMPONENTS,
    masks: list[str] = PRIMARY_MASKS,
    protocol: str = "prespecified_2x2x2_with_mse_on",
) -> list[dict[str, Any]]:
    metrics = [name for name in (
        "mse", "r2", "pearson", "qrs_correlation", "st_correlation",
        "j_point_amplitude_mae", "ecgfounder_auroc",
    ) if name in records]
    effects: list[dict[str, Any]] = []
    component_names = tuple(components)
    effect_type = {1: "main", 2: "pairwise", 3: "three_way", 4: "four_way"}
    definitions = [
        (
            tuple(components[name] for name in names),
            ":".join(names),
            effect_type[order],
        )
        for order in range(1, len(COMPONENTS) + 1)
        for names in combinations(component_names, order)
    ]
    for family in FAMILIES:
        family_frame = records[records["family"] == family]
        for metric in metrics:
            for indices, name, effect_type in definitions:
                contrast = contrast_frame(
                    family_frame, indices, metric, masks=masks
                )
                interval = cluster_bca_mean(
                    contrast["value"].to_numpy(),
                    contrast["patient_id"].to_numpy(),
                )
                effects.append({
                    "family": family,
                    "metric": metric,
                    "effect": name,
                    "effect_type": effect_type,
                    "protocol": protocol,
                    "bootstrap_unit": "paired_ptbxl_patient_cluster",
                    "endpoint_representation": (
                        "macro_auroc_record_influence"
                        if metric == "ecgfounder_auroc"
                        else "record_metric"
                    ),
                    **interval,
                })
    return effects


def familywise_endpoint_tests(records: pd.DataFrame) -> list[dict[str, Any]]:
    """Paired full-vs-base tests for the three prespecified paper endpoints."""
    endpoints = {
        "QRS": "qrs_correlation",
        "ST": "st_correlation",
        "diagnostic_utility": "ecgfounder_auroc",
    }
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        family_frame = records[records["family"] == family]
        for endpoint_index, (endpoint, metric) in enumerate(endpoints.items()):
            pivot = family_frame.pivot(index="ecg_id", columns="mask", values=metric)
            if not {"1000", "1111"}.issubset(pivot.columns):
                differences = np.asarray([], dtype=np.float64)
                difference_ecg_ids = np.asarray([], dtype=np.int64)
            else:
                difference_series = pivot["1111"] - pivot["1000"]
                valid = np.isfinite(difference_series.to_numpy(dtype=np.float64))
                differences = difference_series.to_numpy(dtype=np.float64)[valid]
                difference_ecg_ids = difference_series.index.to_numpy()[valid]
            patient_ids = (
                patient_ids_for_ecgs(difference_ecg_ids)
                if differences.size else np.asarray([], dtype=np.int64)
            )
            interval = cluster_bca_mean(differences, patient_ids)
            p_value = None
            if differences.size >= 3:
                observed = float(differences.mean())
                centered = differences - observed
                cluster_frame = pd.DataFrame({
                    "value": centered,
                    "patient_id": patient_ids,
                })
                grouped = cluster_frame.groupby("patient_id")["value"].agg(
                    ["sum", "count"]
                )
                cluster_sums = grouped["sum"].to_numpy(dtype=np.float64)
                cluster_counts = grouped["count"].to_numpy(dtype=np.float64)
                n_patients = len(grouped)
                rng = np.random.default_rng(4200 + family_index * 10 + endpoint_index)
                indices = rng.integers(
                    0,
                    n_patients,
                    size=(2000, n_patients),
                    dtype=np.int32,
                )
                null_means = (
                    cluster_sums[indices].sum(axis=1)
                    / cluster_counts[indices].sum(axis=1)
                )
                p_value = float(
                    (1 + np.count_nonzero(np.abs(null_means) >= abs(observed)))
                    / 2001
                )
            rows.append({
                "family": family,
                "comparison": "full_1111_minus_base_1000",
                "endpoint": endpoint,
                "metric": metric,
                "alpha": 0.0167,
                "multiplicity_family": (
                    "three prespecified endpoints within this architecture"
                ),
                "p_value": p_value,
                "reject_null": bool(p_value is not None and p_value < 0.0167),
                "bootstrap_unit": "paired_ptbxl_patient_cluster",
                **interval,
            })
    return rows


def summary_table(
    results: dict[str, Any],
    registry: dict[str, Any],
    *,
    repaired_msvae_mse_toggle: bool = False,
) -> pd.DataFrame:
    specs = {spec["id"]: spec for spec in registry["models"]}
    rows = []
    for model_id, model in results["models"].items():
        missing = model["signal"]["missing_leads"]
        clinical = model.get("clinical", {})
        morphology = model.get("morphology", {})
        factorial_mask = specs[model_id]["factorial_mask"]
        rows.append({
            "id": model_id,
            "family": model_id.split("__", 1)[0],
            "mask": factorial_mask,
            "inference_status": (
                "invalid_msvae_mse_toggle_quarantined"
                if (
                    not repaired_msvae_mse_toggle
                    and model_id.startswith("msvae__")
                    and factorial_mask.startswith("0")
                )
                else "valid"
            ),
            "mse": missing.get("mse"),
            "r2": missing.get("r2"),
            "pearson": missing.get("pearson"),
            "qrs_correlation": morphology.get("qrs_correlation"),
            "st_correlation": morphology.get("st_correlation"),
            "ecgfounder_auroc": clinical.get("macro_auroc"),
            "morphology_coverage": morphology.get("detector_coverage", {}).get("fraction"),
        })
    return pd.DataFrame(rows)


def paper_parity_table(results: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    for model_id, model in (results or {}).get("models", {}).items():
        match = re.search(r"__e([01])c([01])m([01])d([01])__s\d+$", model_id)
        if match is None:
            raise ValueError(f"Cannot parse paper-parity factorial mask from {model_id}")
        mask = "".join(match.groups())
        missing = model["signal"]["missing_leads"]
        morphology = model.get("morphology", {})
        clinical = model.get("clinical", {})
        rows.append({
            "id": model_id,
            "mask": mask,
            "loss_protocol": "paper_parity_normalized",
            "n_samples": model["signal"].get("n_samples"),
            "mse": missing.get("mse"),
            "r2": missing.get("r2"),
            "pearson": missing.get("pearson"),
            "qrs_correlation": morphology.get("qrs_correlation"),
            "st_correlation": morphology.get("st_correlation"),
            "ecgfounder_auroc": clinical.get("macro_auroc"),
        })
    return pd.DataFrame(rows)


def heatmaps(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    repaired_msvae_mse_toggle: bool = False,
) -> None:
    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = table[table["mask"].str.startswith("1")].copy()
    primary["mask"] = primary["mask"].str[1:]
    metrics = [("r2", "Missing-lead R²"), ("qrs_correlation", "QRS correlation"), ("st_correlation", "ST correlation"), ("ecgfounder_auroc", "ECGFounder AUROC")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for label, axis, (metric, title) in zip(("A", "B", "C", "D"), axes.ravel(), metrics):
        matrix = primary.pivot(index="family", columns="mask", values=metric).reindex(
            index=FAMILIES, columns=[f"{value:03b}" for value in range(8)]
        )
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Correlation / MMD / derivative mask (MSE on)")
        axis.set_ylabel("")
        axis.set_yticklabels([FAMILY_LABELS.get(value.get_text(), value.get_text()) for value in axis.get_yticklabels()])
        axis.text(-0.08, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(figure_dir / "factorial_heatmaps_mse_on_slice.png", dpi=300)
    fig.savefig(figure_dir / "factorial_heatmaps_mse_on_slice.svg")
    fig.savefig(figure_dir / "factorial_heatmaps_mse_on_slice.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 8), constrained_layout=True)
    supplemental = (
        table.copy()
        if repaired_msvae_mse_toggle
        else table[
            ~((table["family"] == "msvae") & table["mask"].str.startswith("0"))
        ].copy()
    )
    for label, axis, (metric, title) in zip(("A", "B", "C", "D"), axes.ravel(), metrics):
        matrix = supplemental.pivot(index="family", columns="mask", values=metric).reindex(
            index=FAMILIES, columns=MASKS
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            linewidths=0.25,
            linecolor="white",
            cmap="viridis",
            ax=axis,
        )
        axis.set_title(
            f"{title} — complete repaired 2⁴"
            if repaired_msvae_mse_toggle
            else f"{title} — supplemental 2⁴ (MSVAE MSE-off quarantined)"
        )
        axis.set_xlabel("MSE / correlation / MMD / derivative mask")
        axis.set_ylabel("")
        axis.set_yticklabels([FAMILY_LABELS.get(value.get_text(), value.get_text()) for value in axis.get_yticklabels()])
        axis.text(-0.08, 1.04, label, transform=axis.transAxes, fontweight="bold")
    # The complete four-factor view is the primary poster figure. Keep the
    # historical supplementary filename as a byte-identical compatibility copy.
    for stem in ("factorial_heatmaps", "factorial_heatmaps_4factor_supplementary"):
        fig.savefig(figure_dir / f"{stem}.png", dpi=300)
        fig.savefig(figure_dir / f"{stem}.svg")
        fig.savefig(figure_dir / f"{stem}.pdf")
    plt.close(fig)


def forest_plot(effects: pd.DataFrame, output_dir: Path) -> None:
    frame = effects[(effects["metric"] == "r2") & (effects["effect_type"] == "main")].dropna(subset=["estimate"])
    frame = frame.assign(
        family_order=pd.Categorical(frame["family"], categories=FAMILIES, ordered=True),
        effect_order=pd.Categorical(
            frame["effect"],
            categories=("mse", "correlation", "mmd", "derivative"),
            ordered=True,
        ),
    ).sort_values(["family_order", "effect_order"])
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for index, row in frame.reset_index(drop=True).iterrows():
        axis.errorbar(
            row["estimate"],
            index,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o",
            color=FAMILY_COLORS[row["family"]],
            capsize=3,
            markersize=5,
        )
    for start in range(0, len(frame), 4):
        if (start // 4) % 2 == 0:
            axis.axhspan(start - 0.5, start + 3.5, color="#F3F4F6", zorder=-2)
    axis.set_yticks(
        range(len(frame)),
        [
            f"{FAMILY_LABELS[row.family]} · "
            f"{'MMD' if str(row.effect) == 'mmd' else 'MSE' if str(row.effect) == 'mse' else str(row.effect).replace('_', ' ').title()}"
            for row in frame.itertuples()
        ],
    )
    axis.axvline(0, color="#333333", linewidth=0.9, linestyle="--")
    axis.set_xlabel("Paired marginal effect on missing-lead R² (95% BCa CI)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "plots/factorial_effect_forest.png", dpi=300)
    fig.savefig(output_dir / "plots/factorial_effect_forest.svg")
    fig.savefig(output_dir / "plots/factorial_effect_forest.pdf")
    plt.close(fig)


def pareto_plot(table: pd.DataFrame, output_dir: Path) -> None:
    table = table.copy()
    mask_markers = {
        "000": "o",
        "001": "s",
        "010": "^",
        "011": "D",
        "100": "P",
        "101": "X",
        "110": "v",
        "111": "*",
    }
    fig, axis = plt.subplots(figsize=(9.4, 6))
    for family in FAMILIES:
        subset = table[table.family == family].dropna(
            subset=["qrs_correlation", "ecgfounder_auroc"]
        )
        for row in subset.itertuples():
            axis.scatter(
                row.qrs_correlation,
                row.ecgfounder_auroc,
                facecolor=FAMILY_COLORS[family] if row.mask[0] == "1" else "none",
                edgecolor=FAMILY_COLORS[family],
                s=95 if row.mask == "1111" else 55,
                marker=mask_markers[row.mask[1:]],
                alpha=0.9,
                linewidth=1.0,
            )
        ordered = subset.sort_values("qrs_correlation", ascending=False)
        frontier = ordered[
            ordered.ecgfounder_auroc
            >= ordered.ecgfounder_auroc.cummax().shift(fill_value=-np.inf)
        ].sort_values("qrs_correlation")
        axis.plot(
            frontier.qrs_correlation,
            frontier.ecgfounder_auroc,
            color=FAMILY_COLORS[family],
            linewidth=1.1,
            alpha=0.65,
        )
    axis.set_xlabel("QRS correlation")
    axis.set_ylabel("ECGFounder macro AUROC")
    axis.grid(alpha=0.2)
    architecture_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=FAMILY_COLORS[family], markeredgecolor="white",
            markersize=8, label=FAMILY_LABELS[family],
        )
        for family in FAMILIES
    ]
    mask_handles = [
        plt.Line2D(
            [0], [0], marker=marker, linestyle="none",
            color="#4B5563", markersize=7, label=mask,
        )
        for mask, marker in mask_markers.items()
    ]
    architecture_legend = axis.legend(
        handles=architecture_handles,
        frameon=False,
        title="Architecture",
        loc="upper left",
    )
    axis.add_artist(architecture_legend)
    component_legend = axis.legend(
        handles=mask_handles,
        frameon=False,
        title="Corr / MMD / deriv.",
        ncol=2,
        loc="lower right",
    )
    axis.add_artist(component_legend)
    axis.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", linestyle="none", color="#4B5563",
                markerfacecolor="#4B5563", markersize=7, label="MSE on",
            ),
            plt.Line2D(
                [0], [0], marker="o", linestyle="none", color="#4B5563",
                markerfacecolor="none", markersize=7, label="MSE off",
            ),
        ],
        frameon=False,
        title="MSE factor",
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "plots/morphology_diagnostic_pareto.png", dpi=300)
    fig.savefig(output_dir / "plots/morphology_diagnostic_pareto.svg")
    fig.savefig(output_dir / "plots/morphology_diagnostic_pareto.pdf")
    plt.close(fig)


def degradation_plot(results: dict[str, Any], registry: dict[str, Any], output_dir: Path) -> None:
    specs = {spec["id"]: spec for spec in registry["models"]}
    rows = []
    for model_id, model in results["models"].items():
        mask = specs[model_id]["factorial_mask"]
        for condition_id, condition in model["robustness"]["conditions"].items():
            if condition["source"] == "NSTDB":
                rows.append({"family": model_id.split("__", 1)[0], "mask": mask, "noise": condition["noise_type"], "snr_db": condition["snr_db"], "mse": condition["signal"]["missing_leads"]["mse"]})
    frame = pd.DataFrame(rows)
    plot = sns.relplot(data=frame, x="snr_db", y="mse", hue="mask", col="family", row="noise", kind="line", marker="o", facet_kws={"sharey": False}, height=3)
    plot.set_axis_labels("SNR (dB)", "Missing-lead MSE")
    plot.savefig(output_dir / "plots/nstdb_degradation.png", dpi=300)
    plot.savefig(output_dir / "plots/nstdb_degradation.svg")
    plot.savefig(output_dir / "plots/nstdb_degradation.pdf")
    plt.close("all")


def echonext_plots(echonext: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for model_id, model in echonext.get("models", {}).items():
        missing = model["signal"]["missing_leads"]
        rows.append({
            "id": model_id,
            "family": model_id.split("__", 1)[0],
            "mask": model["factorial_mask"],
            "r2": missing.get("r2"),
            "pearson": missing.get("pearson"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("r2", "pearson"),
        ("EchoNext missing-lead R²", "EchoNext missing-lead Pearson"),
    ):
        matrix = frame.pivot(index="family", columns="mask", values=metric).reindex(
            index=FAMILIES, columns=MASKS
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            linewidths=0.25,
            linecolor="white",
            cmap="viridis",
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("MSE / correlation / MMD / derivative mask")
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.savefig(figure_dir / "echonext_external_heatmap.png", dpi=300)
    fig.savefig(figure_dir / "echonext_external_heatmap.svg")
    fig.savefig(figure_dir / "echonext_external_heatmap.pdf")
    plt.close(fig)

    selected = [
        echonext["models"].get(f"{family}__e1c1m1d1__s42") for family in FAMILIES
    ]
    if all(model and model.get("example_plot") for model in selected):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        for label, axis, family, model in zip(("A", "B", "C"), axes, FAMILIES, selected):
            image_path = PROJECT_ROOT / model["example_plot"]
            axis.imshow(plt.imread(image_path))
            axis.set_title(f"{FAMILY_LABELS[family]} · full objective")
            axis.axis("off")
            axis.text(0.01, 0.99, label, transform=axis.transAxes, fontweight="bold", va="top")
        fig.savefig(figure_dir / "echonext_representative_reconstructions.png", dpi=300)
        fig.savefig(figure_dir / "echonext_representative_reconstructions.svg")
        fig.savefig(figure_dir / "echonext_representative_reconstructions.pdf")
        plt.close(fig)


def echonext_shd_label_audit(echonext: dict[str, Any] | None) -> dict[str, Any]:
    """Verify identical official SHD labels in reference and reconstructed tables."""
    if not echonext:
        return {
            "status": "not_available",
            "prediction_metrics_status": "not_computed",
        }
    reference_path = PROJECT_ROOT / echonext.get("shd_reference", {}).get(
        "per_record_parquet", "__missing__"
    )
    if not reference_path.is_file():
        return {
            "status": "fail",
            "reason": "missing_official_reference_shd_parquet",
            "path": str(reference_path),
        }
    reference = pd.read_parquet(
        reference_path, columns=["row_index", "ecg_key", "labels"]
    )
    reference_labels = np.stack(reference["labels"].map(np.asarray).to_numpy())
    if reference_labels.shape != (5442, len(ECHONEXT_SHD_LABELS)):
        return {
            "status": "fail",
            "reason": "invalid_official_reference_label_shape",
            "shape": list(reference_labels.shape),
        }
    reference_frame = pd.DataFrame(
        reference_labels, columns=ECHONEXT_SHD_LABELS
    )
    reference_frame.insert(0, "ecg_key", reference["ecg_key"].to_numpy())
    reference_frame.insert(0, "row_index", reference["row_index"].to_numpy())
    reference_hash = hashlib.sha256(
        pd.util.hash_pandas_object(reference_frame, index=True).values.tobytes()
    ).hexdigest()
    model_checks: dict[str, Any] = {}
    label_summary = {
        label: {
            "nonmissing": int(reference_frame[label].notna().sum()),
            "positive": int((reference_frame[label] == 1).sum()),
            "negative": int((reference_frame[label] == 0).sum()),
        }
        for label in ECHONEXT_SHD_LABELS
    }
    for model_id, model in echonext.get("models", {}).items():
        path = PROJECT_ROOT / model.get(
            "shd_per_record_parquet", "__missing__"
        )
        if not path.exists():
            model_checks[model_id] = {"status": "missing", "path": str(path)}
            continue
        frame = pd.read_parquet(
            path, columns=["row_index", "ecg_key", "condition", "labels"]
        )
        frame = frame.loc[frame["condition"] == "clean"].reset_index(drop=True)
        try:
            labels = np.stack(frame["labels"].map(np.asarray).to_numpy())
        except ValueError:
            labels = np.empty((0, len(ECHONEXT_SHD_LABELS)))
        if labels.ndim == 2 and labels.shape[1] == len(ECHONEXT_SHD_LABELS):
            expanded = pd.DataFrame(labels, columns=ECHONEXT_SHD_LABELS)
            expanded.insert(0, "ecg_key", frame["ecg_key"].to_numpy())
            expanded.insert(0, "row_index", frame["row_index"].to_numpy())
        else:
            expanded = pd.DataFrame()
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(expanded, index=True).values.tobytes()
        ).hexdigest()
        model_checks[model_id] = {
            "status": (
                "pass"
                if labels.shape == (5442, len(ECHONEXT_SHD_LABELS))
                and digest == reference_hash
                else "fail"
            ),
            "records": len(frame),
            "label_shape": list(labels.shape),
            "label_hash": digest,
        }
    passed = (
        len(model_checks) == 48
        and all(check["status"] == "pass" for check in model_checks.values())
    )
    return {
        "status": "pass" if passed else "fail",
        "labels": list(ECHONEXT_SHD_LABELS),
        "records_per_model": 5442,
        "label_hash": reference_hash,
        "label_summary": label_summary,
        "model_checks": model_checks,
        "prediction_metrics_status": "computed_with_official_frozen_echonext_minimodel",
        "interpretation": (
            "Echocardiography-derived SHD labels are identity-checked across "
            "the official reference and every reconstructed clean table."
        ),
    }


def completeness(
    results: dict[str, Any],
    registry: dict[str, Any],
    echonext: dict[str, Any] | None,
    echonext_shd_audit: dict[str, Any],
    smartwatch: dict[str, Any] | None,
    paper_parity: dict[str, Any] | None,
    confirmations: pd.DataFrame,
    noninferiority: pd.DataFrame,
) -> dict[str, Any]:
    expected = {spec["id"] for spec in registry["models"]}
    actual = set(results.get("models", {}))
    expected_conditions = {
        "gaussian_24db", "gaussian_12db", "gaussian_6db", "gaussian_0db",
        "baseline_drift",
        "nstdb_bw_24db", "nstdb_bw_12db", "nstdb_bw_6db", "nstdb_bw_0db",
        "nstdb_em_24db", "nstdb_em_12db", "nstdb_em_6db", "nstdb_em_0db",
        "nstdb_ma_24db", "nstdb_ma_12db", "nstdb_ma_6db", "nstdb_ma_0db",
    }
    optional_fitbit_conditions = {
        "fitbit_20db", "fitbit_10db", "fitbit_baseline_wander",
    }
    condition_sets = []
    samples = []
    stress_samples = []
    clinical_complete = True
    morphology_complete = True
    strict_record_paths = True
    required_clinical = {
        "macro_auroc", "macro_average_precision", "macro_f1",
        "macro_sensitivity", "macro_specificity", "calibration",
    }
    for model in results.get("models", {}).values():
        robustness = model.get("robustness", {})
        condition_sets.append(set(robustness.get("conditions", {})))
        samples.append(model.get("signal", {}).get("n_samples"))
        stress_samples.append(robustness.get("n_samples"))
        clinical = model.get("clinical", {})
        clinical_complete &= required_clinical.issubset(clinical)
        clinical_complete &= safe_float(clinical.get("macro_auroc")) is not None
        clinical_complete &= "gaps" in model.get("fairness", {})
        clinical_complete &= safe_float(
            model.get("paper_parity_clinical", {}).get("macro_auroc")
        ) is not None
        morphology = model.get("morphology", {})
        morphology_complete &= "detector_coverage" in morphology
        morphology_complete &= "spectral_relative_error" in morphology
        morphology_complete &= "class_stratified_qrs_correlation" in morphology
        strict_record_paths &= (PROJECT_ROOT / model.get("per_record_parquet", "__missing__")).exists()
        strict_record_paths &= (PROJECT_ROOT / model.get("ecgfounder_per_record_parquet", "__missing__")).exists()
        strict_record_paths &= (PROJECT_ROOT / model.get("paper_parity_per_record_parquet", "__missing__")).exists()
    echo_models = set(echonext.get("models", {})) if echonext else set()
    echo_payloads = list(echonext.get("models", {}).values()) if echonext else []
    echo_record_paths = bool(echo_payloads) and all(
        (PROJECT_ROOT / model.get("per_record_parquet", "__missing__")).exists()
        and (PROJECT_ROOT / model.get("shd_per_record_parquet", "__missing__")).exists()
        for model in echo_payloads
    )
    echo_conditions = [
        set(model.get("robustness", {}).get("conditions", {}))
        for model in echo_payloads
    ]
    smartwatch_models = set(smartwatch.get("models", {})) if smartwatch else set()
    expected_smartwatch_counts = {
        "applewatch_serie8": 180,
        "fitbitsense2": 180,
        "samsunggalaxy6": 179,
        "withingsscanwatch": 180,
    }
    smartwatch_clean_complete = bool(smartwatch_models) and all(
        set(devices) == set(expected_smartwatch_counts)
        and all(
            devices[device].get("n_paired_records") == expected_count
            and devices[device].get("reconstruction_passthrough_leads") == ["II"]
            and devices[device]
            .get("signal_wearable_missing_eleven", {})
            .get("missing_leads", {})
            .get("names")
            == ["I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            and devices[device]
            .get("ecgfounder_probability_fidelity", {})
            .get("n_tasks")
            == 150
            and (
                PROJECT_ROOT
                / devices[device].get("per_record_parquet", "__missing__")
            ).is_file()
            for device, expected_count in expected_smartwatch_counts.items()
        )
        for devices in smartwatch.get("models", {}).values()
    )
    paper_models = list((paper_parity or {}).get("models", {}).values())
    confirmation_configurations = (
        confirmations[["family", "slot"]].drop_duplicates().shape[0]
        if not confirmations.empty else 0
    )
    checks = {
        "primary_cells_48": len(expected) == 48 and actual == expected,
        "confirmation_configurations_9": confirmation_configurations == 9,
        "confirmation_runs_18": len(confirmations) == 18,
        "confirmation_validation_records_2183": bool(len(confirmations) == 18 and (confirmations["n_samples"] == 2183).all()),
        "clean_records_2198": len(samples) == 48 and set(samples) == {2198},
        "stress_records_2198": len(stress_samples) == 48 and set(stress_samples) == {2198},
        "core_stress_conditions_17": len(condition_sets) == 48 and all(
            expected_conditions.issubset(item)
            and item.issubset(expected_conditions | optional_fitbit_conditions)
            for item in condition_sets
        ),
        "fitbit_conditions_all_or_none": len(condition_sets) == 48 and all(
            (item & optional_fitbit_conditions) in (set(), optional_fitbit_conditions)
            for item in condition_sets
        ),
        "ecgfounder_complete": clinical_complete,
        "ecgfounder_noninferiority_tested_48": len(noninferiority) == 48,
        "ecgfounder_noninferiority_results_finite_48": bool(
            len(noninferiority) == 48
            and {
                "reconstructed_macro_auroc",
                "reference_macro_auroc",
                "difference",
                "ci_low",
                "ci_high",
                "margin",
                "noninferior",
            }.issubset(noninferiority.columns)
            and noninferiority[
                [
                    "reconstructed_macro_auroc",
                    "reference_macro_auroc",
                    "difference",
                    "ci_low",
                    "ci_high",
                    "margin",
                    "noninferior",
                ]
            ].notna().all().all()
        ),
        "morphology_coverage_reported": morphology_complete,
        "per_record_artifacts_present": strict_record_paths,
        "paper_parity_anchors_5": len(paper_models) == 5,
        "paper_parity_records_2198": bool(
            len(paper_models) == 5
            and all(model.get("signal", {}).get("n_samples") == 2198 for model in paper_models)
        ),
        "paper_parity_metrics_complete": bool(
            len(paper_models) == 5
            and all(
                safe_float(model.get("clinical", {}).get("macro_auroc")) is not None
                and "detector_coverage" in model.get("morphology", {})
                for model in paper_models
            )
        ),
        "echonext_cells_48": echo_models == expected,
        "echonext_clean_records_5442": bool(
            len(echo_payloads) == 48
            and all(model.get("signal", {}).get("n_samples") == 5442 for model in echo_payloads)
        ),
        "echonext_morphology_records_2198": bool(
            len(echo_payloads) == 48
            and all(model.get("morphology_evaluated_records") == 2198 for model in echo_payloads)
        ),
        "echonext_stress_records_2198": bool(
            len(echo_payloads) == 48
            and all(model.get("robustness", {}).get("n_samples") == 2198 for model in echo_payloads)
        ),
        "echonext_stress_conditions_17": bool(
            len(echo_conditions) == 48
            and all(conditions == expected_conditions for conditions in echo_conditions)
        ),
        "echonext_per_record_artifacts_present": echo_record_paths,
        "echonext_shd_labels_attached": echonext_shd_audit.get("status") == "pass",
        "smartwatch_cells_48": smartwatch_models == expected,
        "smartwatch_protocol_complete": bool(
            smartwatch
            and set(smartwatch.get("protocol", {}).get("conditions", []))
            == {"amp_test", "freq_test", "sqr-2hz", "st-segment"}
            and smartwatch.get("protocol", {}).get("extra_injected_stress") == "not_run"
            and smartwatch.get("protocol", {})
            .get("evaluation_taxonomy", {})
            .get("human_diagnostic_ground_truth")
            == "unavailable"
            and set(smartwatch.get("protocol", {}).get("pairing_audit", {}))
            == set(expected_smartwatch_counts)
            and all(
                audit.get("pairing_key") == "casefolded_relative_posix_path"
                and audit.get("paired_records") == expected_smartwatch_counts[device]
                for device, audit in smartwatch.get("protocol", {})
                .get("pairing_audit", {})
                .items()
            )
            and smartwatch_clean_complete
        ),
    }
    return {"checks": checks, "status": "complete" if all(checks.values()) else "incomplete"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--echonext", type=Path)
    parser.add_argument("--smartwatch", type=Path)
    parser.add_argument("--paper-parity", type=Path)
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        help=(
            "Directory containing the 18 confirmation-run marker JSON files. "
            "Defaults to <output-dir>/confirmation."
        ),
    )
    parser.add_argument(
        "--msvae-mse-repair-attestation",
        type=Path,
        help=(
            "PASS attestation required to promote repaired MultiScale-VAE "
            "MSE-off cells into the complete 2^4 analysis"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    registry_path = args.registry if args.registry.is_absolute() else PROJECT_ROOT / args.registry
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    confirmation_dir = (
        args.confirmation_dir
        if args.confirmation_dir and args.confirmation_dir.is_absolute()
        else PROJECT_ROOT / args.confirmation_dir
        if args.confirmation_dir
        else output_dir / "confirmation"
    )
    results = json.loads(input_path.read_text())
    registry = json.loads(registry_path.read_text())
    paper_parity_path = (
        args.paper_parity
        if args.paper_parity and args.paper_parity.is_absolute()
        else PROJECT_ROOT / args.paper_parity
        if args.paper_parity
        else None
    )
    paper_parity = json.loads(paper_parity_path.read_text()) if paper_parity_path else None
    repair_attestation_path = (
        args.msvae_mse_repair_attestation
        if args.msvae_mse_repair_attestation
        and args.msvae_mse_repair_attestation.is_absolute()
        else PROJECT_ROOT / args.msvae_mse_repair_attestation
        if args.msvae_mse_repair_attestation
        else None
    )
    repair_attestation = (
        json.loads(repair_attestation_path.read_text())
        if repair_attestation_path
        else None
    )
    repaired_msvae_mse_toggle = bool(
        repair_attestation and repair_attestation.get("status") == "PASS"
    )
    if repair_attestation_path and not repaired_msvae_mse_toggle:
        raise ValueError("MultiScale-VAE MSE repair attestation is not PASS")
    if len(registry["models"]) not in (24, 48):
        print(f"Warning: Registry contains {len(registry['models'])} models (expected 24 or 48). Proceeding with analysis.")
    table = summary_table(
        results,
        registry,
        repaired_msvae_mse_toggle=repaired_msvae_mse_toggle,
    )
    parity_table = paper_parity_table(paper_parity)
    records = load_record_table(results, registry)
    effects = pd.DataFrame(factorial_effects(records))
    expanded_effects = pd.DataFrame(
        factorial_effects(
            records,
            components=COMPONENTS,
            masks=MASKS,
            protocol=(
                "primary_repaired_2x2x2x2_including_mse_toggle"
                if repaired_msvae_mse_toggle
                else "supplemental_2x2x2x2_including_mse_toggle"
            ),
        )
    )
    if not repaired_msvae_mse_toggle:
        expanded_effects = expanded_effects[
            expanded_effects["family"] != "msvae"
        ].reset_index(drop=True)
    claim_effects = expanded_effects if repaired_msvae_mse_toggle else effects
    endpoint_tests = pd.DataFrame(familywise_endpoint_tests(records))
    confirmations = confirmation_table(confirmation_dir)
    noninferiority = pd.DataFrame(ecgfounder_noninferiority(results, registry))
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "main_48_cell_table.csv", index=False)
    table.to_parquet(output_dir / "main_48_cell_table.parquet", index=False)
    table.to_html(output_dir / "main_48_cell_table.html", index=False, float_format=lambda x: f"{x:.4f}")
    table.to_latex(output_dir / "main_48_cell_table.tex", index=False, float_format="%.4f")
    primary_table = table[table["mask"].str.startswith("1")].copy()
    primary_table["mask"] = primary_table["mask"].str[1:]
    primary_table.to_csv(output_dir / "main_primary_24_cell_table.csv", index=False)
    primary_table.to_parquet(output_dir / "main_primary_24_cell_table.parquet", index=False)
    primary_table.to_html(
        output_dir / "main_primary_24_cell_table.html",
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
    primary_table.to_latex(
        output_dir / "main_primary_24_cell_table.tex",
        index=False,
        float_format="%.4f",
    )
    parity_table.to_csv(output_dir / "paper_parity_anchor_table.csv", index=False)
    parity_table.to_parquet(output_dir / "paper_parity_anchor_table.parquet", index=False)
    parity_table.to_latex(
        output_dir / "paper_parity_anchor_table.tex", index=False, float_format="%.4f"
    )
    claim_effects.to_csv(output_dir / "factorial_effects_bca.csv", index=False)
    claim_effects.to_parquet(output_dir / "factorial_effects_bca.parquet", index=False)
    effects.to_csv(
        output_dir / "factorial_effects_mse_on_conditional.csv", index=False
    )
    effects.to_parquet(
        output_dir / "factorial_effects_mse_on_conditional.parquet", index=False
    )
    expanded_effects.to_csv(
        output_dir / "factorial_effects_4factor_supplementary.csv", index=False
    )
    expanded_effects.to_parquet(
        output_dir / "factorial_effects_4factor_supplementary.parquet", index=False
    )
    endpoint_tests.to_csv(output_dir / "familywise_endpoint_tests.csv", index=False)
    endpoint_tests.to_parquet(output_dir / "familywise_endpoint_tests.parquet", index=False)
    confirmations.to_csv(output_dir / "supplementary_seed_table.csv", index=False)
    confirmations.to_parquet(output_dir / "supplementary_seed_table.parquet", index=False)
    confirmations.to_latex(
        output_dir / "supplementary_seed_table.tex", index=False, float_format="%.4f"
    )
    noninferiority.to_csv(output_dir / "ecgfounder_noninferiority.csv", index=False)
    noninferiority.to_parquet(output_dir / "ecgfounder_noninferiority.parquet", index=False)
    strict_write(output_dir / "statistics.json", {
        "bootstrap_resamples": 2000,
        "interval": "BCa 95%",
        "primary_factorial_protocol": (
            "complete repaired 2^4 MSE/correlation/MMD/derivative factorial"
            if repaired_msvae_mse_toggle
            else "2^3 correlation/MMD/derivative with MSE enabled"
        ),
        "effects": claim_effects.to_dict(orient="records"),
        "conditional_mse_on_effects": effects.to_dict(orient="records"),
        "supplemental_factorial_protocol": (
            "not applicable; repaired 2^4 promoted by PASS attestation"
            if repaired_msvae_mse_toggle
            else (
                "2^4 exploratory MSE toggle for U-Net and ECG-AIM only; "
                "MultiScale-VAE MSE-toggle cells are quarantined because "
                "detached decoder loss prevented lambda_mse from changing "
                "gradients."
            )
        ),
        "supplemental_exclusions": (
            {}
            if repaired_msvae_mse_toggle
            else {
                "msvae_mse_toggle": (
                    "invalid; all eight matched e0/e1 checkpoint pairs are "
                    "tensor-identical"
                )
            }
        ),
        "msvae_mse_repair_attestation_sha256": (
            file_sha256(repair_attestation_path)
            if repair_attestation_path
            else None
        ),
        "supplemental_effects": expanded_effects.to_dict(orient="records"),
        "familywise_alpha": 0.0167,
        "familywise_endpoints": ["QRS", "ST", "diagnostic_utility"],
        "familywise_scope": (
            "Bonferroni control across the three prespecified endpoints within "
            "each architecture; not across all nine architecture-endpoint tests."
        ),
        "factorial_effect_ci_scope": (
            "The 147 primary factorial-effect intervals are unadjusted, "
            "descriptive patient-cluster BCa intervals; no cross-family "
            "any-effect claim is licensed."
        ),
        "familywise_endpoint_tests": endpoint_tests.to_dict(orient="records"),
        "ecgfounder_noninferiority_margin": ECGFOUNDER_NI_MARGIN,
        "ecgfounder_noninferiority_scope": (
            "Per-cell, pointwise comparison against a prespecified engineering "
            "margin of 0.02; the margin is not a clinically validated "
            "equivalence threshold and the 48 intervals are not multiplicity "
            "adjusted. Influence-function intervals are asymptotic, including "
            "five evaluable tasks with <=10 positives."
        ),
        "ecgfounder_noninferiority_summary": {
            "tested": int(len(noninferiority)),
            "passed": int(noninferiority["noninferior"].sum()),
            "failed": int((~noninferiority["noninferior"]).sum()),
            "interpretation": (
                f"{int(noninferiority['noninferior'].sum())}/{len(noninferiority)} "
                "reconstructed models meet the prespecified 0.02 margin; "
                + ", ".join(
                    f"{family}: "
                    f"{int(group['noninferior'].sum())}/{len(group)}"
                    for family, group in noninferiority.groupby(
                        "family", sort=True
                    )
                )
                + ". This is an outcome count, not a completeness requirement."
            ),
        },
        "ecgfounder_noninferiority": noninferiority.to_dict(orient="records"),
        "confirmation_runs": confirmations.to_dict(orient="records"),
        "inference_scope": {
            "patient_cluster_bca": (
                "Conditional on each trained seed-42 model; supports paired "
                "patient-cluster contrasts across 1,904 patients/2,198 ECGs, "
                "not population-level training-seed generalization."
            ),
            "training_variance": (
                "Seeds 1337 and 2026 cover base/full/validation-selected "
                "configurations only and are reported separately."
            ),
            "ecgfounder_macro_auroc": (
                "Computed over the 31/150 fold-10 tasks containing both classes; "
                "119 tasks are explicitly single-class."
            ),
        },
    })
    echonext = json.loads((args.echonext if args.echonext.is_absolute() else PROJECT_ROOT / args.echonext).read_text()) if args.echonext else None
    smartwatch_path = (
        args.smartwatch
        if args.smartwatch and args.smartwatch.is_absolute()
        else PROJECT_ROOT / args.smartwatch
        if args.smartwatch
        else None
    )
    smartwatch = json.loads(smartwatch_path.read_text()) if smartwatch_path else None
    shd_audit = echonext_shd_label_audit(echonext)
    strict_write(output_dir / "echonext_shd_label_audit.json", shd_audit)
    audit = completeness(
        results,
        registry,
        echonext,
        shd_audit,
        smartwatch,
        paper_parity,
        confirmations,
        noninferiority,
    )
    audit["checks"]["factorial_effect_rows_complete"] = len(effects) == (
        len(FAMILIES) * len(records.columns.intersection([
            "mse", "r2", "pearson", "qrs_correlation", "st_correlation",
            "j_point_amplitude_mae", "ecgfounder_auroc",
        ])) * (2 ** len(PRIMARY_COMPONENTS) - 1)
    )
    audit["checks"]["supplemental_4factor_effect_rows_complete_valid_families"] = len(
        expanded_effects
    ) == (
        len(FAMILIES) * len(records.columns.intersection([
            "mse", "r2", "pearson", "qrs_correlation", "st_correlation",
            "j_point_amplitude_mae", "ecgfounder_auroc",
        ])) * (2 ** len(COMPONENTS) - 1)
    )
    audit["checks"]["familywise_endpoint_tests_9"] = len(endpoint_tests) == 9
    audit["status"] = "complete" if all(audit["checks"].values()) else "incomplete"

    plot_paths = {
        "factorial_heatmaps": output_dir / "plots/factorial_heatmaps.png",
        "factorial_effect_forest": output_dir / "plots/factorial_effect_forest.png",
        "morphology_diagnostic_pareto": output_dir / "plots/morphology_diagnostic_pareto.png",
        "nstdb_degradation": output_dir / "plots/nstdb_degradation.png",
        "echonext_external_heatmap": output_dir / "plots/echonext_external_heatmap.png",
        "echonext_representative_reconstructions": output_dir / "plots/echonext_representative_reconstructions.png",
        "factorial_heatmaps_svg": output_dir / "plots/factorial_heatmaps.svg",
        "factorial_heatmaps_pdf": output_dir / "plots/factorial_heatmaps.pdf",
        "factorial_effect_forest_svg": output_dir / "plots/factorial_effect_forest.svg",
        "factorial_effect_forest_pdf": output_dir / "plots/factorial_effect_forest.pdf",
        "morphology_diagnostic_pareto_svg": output_dir / "plots/morphology_diagnostic_pareto.svg",
        "morphology_diagnostic_pareto_pdf": output_dir / "plots/morphology_diagnostic_pareto.pdf",
        "nstdb_degradation_svg": output_dir / "plots/nstdb_degradation.svg",
        "nstdb_degradation_pdf": output_dir / "plots/nstdb_degradation.pdf",
        "echonext_external_heatmap_svg": output_dir / "plots/echonext_external_heatmap.svg",
        "echonext_external_heatmap_pdf": output_dir / "plots/echonext_external_heatmap.pdf",
        "echonext_representative_reconstructions_svg": output_dir / "plots/echonext_representative_reconstructions.svg",
        "echonext_representative_reconstructions_pdf": output_dir / "plots/echonext_representative_reconstructions.pdf",
        "factorial_heatmaps_4factor_supplementary": output_dir / "plots/factorial_heatmaps_4factor_supplementary.png",
        "factorial_heatmaps_4factor_supplementary_svg": output_dir / "plots/factorial_heatmaps_4factor_supplementary.svg",
        "factorial_heatmaps_4factor_supplementary_pdf": output_dir / "plots/factorial_heatmaps_4factor_supplementary.pdf",
        "factorial_heatmaps_mse_on_slice": output_dir / "plots/factorial_heatmaps_mse_on_slice.png",
        "factorial_heatmaps_mse_on_slice_svg": output_dir / "plots/factorial_heatmaps_mse_on_slice.svg",
        "factorial_heatmaps_mse_on_slice_pdf": output_dir / "plots/factorial_heatmaps_mse_on_slice.pdf",
    }
    for path in plot_paths.values():
        path.unlink(missing_ok=True)
    core_ready = all(
        passed
        for name, passed in audit["checks"].items()
        if not name.startswith(("echonext_", "smartwatch_"))
    )
    if core_ready:
        heatmaps(
            table,
            output_dir,
            repaired_msvae_mse_toggle=repaired_msvae_mse_toggle,
        )
        forest_plot(claim_effects, output_dir)
        pareto_plot(table, output_dir)
        degradation_plot(results, registry, output_dir)
    echo_ready = all(
        passed for name, passed in audit["checks"].items() if name.startswith("echonext_")
    )
    if core_ready and echo_ready and echonext:
        echonext_plots(echonext, output_dir)
    strict_write(output_dir / "figure_provenance.json", {
        "source_sha256": {
            "generator_script": file_sha256(Path(__file__).resolve()),
            "comprehensive_results": file_sha256(input_path),
            "model_registry": file_sha256(registry_path),
            "paper_parity_results": file_sha256(paper_parity_path) if paper_parity_path else None,
            "echonext_results": file_sha256(
                args.echonext if args.echonext and args.echonext.is_absolute()
                else PROJECT_ROOT / args.echonext
            ) if args.echonext else None,
            "smartwatch_results": file_sha256(smartwatch_path) if smartwatch_path else None,
            "main_table": file_sha256(output_dir / "main_48_cell_table.csv"),
            "primary_main_table": file_sha256(
                output_dir / "main_primary_24_cell_table.csv"
            ),
            "effects_table": file_sha256(output_dir / "factorial_effects_bca.csv"),
        },
        "core_coverage_passed": core_ready,
        "generated_figures": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in plot_paths.items()
            if path.exists()
        },
    })
    strict_write(output_dir / "completeness.json", audit)
    report = ["# Factorial v2 Completeness Report", "", f"Status: **{audit['status']}**", ""] + [f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in audit["checks"].items()]
    (output_dir / "COMPLETENESS_REPORT.md").write_text("\n".join(report) + "\n")
    if args.require_complete and audit["status"] != "complete":
        raise SystemExit(5)


if __name__ == "__main__":
    main()
