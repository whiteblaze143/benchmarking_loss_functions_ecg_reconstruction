from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

import pandas as pd

from scripts.analyze_factorial_v2 import (
    FAMILIES,
    PRIMARY_MASKS,
    _auc_and_influence,
    _paired_influence_bca,
    bca_mean,
    contrast_vector,
    degradation_plot,
    echonext_plots,
    forest_plot,
    heatmaps,
    pareto_plot,
    summary_table,
)


def test_constant_sample_has_exact_degenerate_bca_interval() -> None:
    interval = bca_mean(np.full(20, 0.125))
    assert interval == {"estimate": 0.125, "ci_low": 0.125, "ci_high": 0.125, "n": 20}


def test_macro_auroc_influence_matches_sklearn_and_is_centered() -> None:
    rng = np.random.default_rng(17)
    labels = rng.integers(0, 2, size=(240, 7), dtype=np.int8)
    probabilities = rng.random((240, 7))
    estimate, influence = _auc_and_influence(labels, probabilities)
    expected = np.mean([
        roc_auc_score(labels[:, task], probabilities[:, task])
        for task in range(labels.shape[1])
    ])
    assert np.isclose(estimate, expected, atol=1e-12)
    assert influence.shape == (labels.shape[0],)
    assert np.isclose(influence.mean(), 0.0, atol=1e-12)


def test_paired_influence_bca_is_deterministic_and_contains_null() -> None:
    rng = np.random.default_rng(3)
    influence = rng.normal(0.0, 0.05, size=300)
    influence -= influence.mean()
    first = _paired_influence_bca(0.0, influence, n_resamples=200, seed=11)
    second = _paired_influence_bca(0.0, influence, n_resamples=200, seed=11)
    assert first == second
    assert first["ci_low"] < 0.0 < first["ci_high"]
    assert first["n_records"] == 300
    assert first["n_resamples"] == 200


def test_identical_predictions_have_zero_paired_interval() -> None:
    rng = np.random.default_rng(9)
    labels = rng.integers(0, 2, size=(200, 5), dtype=np.int8)
    probabilities = rng.random((200, 5))
    auc_a, influence_a = _auc_and_influence(labels, probabilities)
    auc_b, influence_b = _auc_and_influence(labels, probabilities.copy())
    interval = _paired_influence_bca(
        auc_a - auc_b,
        influence_a - influence_b,
        n_resamples=100,
    )
    assert interval["estimate"] == 0.0
    assert interval["ci_low"] == 0.0
    assert interval["ci_high"] == 0.0


def test_four_factor_main_effect_is_average_matched_contrast() -> None:
    rows = []
    for record_id, offset in enumerate((0.0, 7.0)):
        for value in range(16):
            mask = f"{value:04b}"
            rows.append({
                "ecg_id": record_id,
                "mask": mask,
                "metric": offset + 2.5 * int(mask[0]) + 0.1 * int(mask[1]),
            })
    effect = contrast_vector(pd.DataFrame(rows), (0,), "metric")
    assert np.allclose(effect, 2.5)


def test_four_factor_interaction_uses_all_sixteen_cells() -> None:
    rows = []
    for record_id in range(3):
        for value in range(16):
            mask = f"{value:04b}"
            rows.append({
                "ecg_id": record_id,
                "mask": mask,
                "metric": 4.0 * int(mask[1]) * int(mask[3]),
            })
    interaction = contrast_vector(pd.DataFrame(rows), (1, 3), "metric")
    assert np.allclose(interaction, 4.0)


def test_primary_three_factor_effect_holds_mse_on() -> None:
    rows = []
    for record_id in range(2):
        for value in range(16):
            mask = f"{value:04b}"
            rows.append({
                "ecg_id": record_id,
                "mask": mask,
                "metric": (
                    100.0 * int(mask[0])
                    + 3.0 * int(mask[1])
                    + 20.0 * int(mask[0] == "0" and mask[1] == "1")
                ),
            })
    effect = contrast_vector(
        pd.DataFrame(rows), (1,), "metric", masks=PRIMARY_MASKS
    )
    assert np.allclose(effect, 3.0)


def test_repair_attestation_promotes_msvae_mse_off_cell() -> None:
    model_id = "msvae__e0c1m0d0__s42"
    registry = {
        "models": [{"id": model_id, "factorial_mask": "0100"}]
    }
    results = {
        "models": {
            model_id: {
                "signal": {
                    "missing_leads": {
                        "mse": 0.1,
                        "r2": 0.2,
                        "pearson": 0.3,
                    }
                },
                "clinical": {"macro_auroc": 0.8},
                "morphology": {
                    "qrs_correlation": 0.7,
                    "st_correlation": 0.6,
                    "detector_coverage": {"fraction": 1.0},
                },
            }
        }
    }
    quarantined = summary_table(results, registry)
    repaired = summary_table(
        results, registry, repaired_msvae_mse_toggle=True
    )
    assert quarantined.loc[0, "inference_status"].startswith("invalid_")
    assert repaired.loc[0, "inference_status"] == "valid"


def test_core_poster_figures_emit_pdf_svg_png_without_empty_layout(tmp_path) -> None:
    table_rows = []
    for family_index, family in enumerate(FAMILIES):
        for value in range(16):
            mask = f"{value:04b}"
            table_rows.append(
                {
                    "family": family,
                    "mask": mask,
                    "r2": 0.55 + 0.01 * family_index + 0.002 * value,
                    "qrs_correlation": 0.70 + 0.001 * value,
                    "st_correlation": 0.65 + 0.001 * value,
                    "ecgfounder_auroc": 0.78 + 0.001 * value,
                }
            )
    table = pd.DataFrame(table_rows)
    effects = pd.DataFrame(
        [
            {
                "family": family,
                "metric": "r2",
                "effect_type": "main",
                "effect": effect,
                "estimate": 0.01 * (effect_index + 1),
                "ci_low": 0.005 * (effect_index + 1),
                "ci_high": 0.015 * (effect_index + 1),
            }
            for family in FAMILIES
            for effect_index, effect in enumerate(("correlation", "mmd", "derivative"))
        ]
    )
    registry_models = []
    result_models = {}
    echo_models = {}
    for family in FAMILIES:
        for value in range(8):
            component_mask = f"{value:03b}"
            mask = f"1{component_mask}"
            model_id = f"{family}__e1c{component_mask[0]}m{component_mask[1]}d{component_mask[2]}__s42"
            registry_models.append({"id": model_id, "factorial_mask": mask})
            conditions = {
                f"nstdb_{noise}_{snr}db": {
                    "source": "NSTDB",
                    "noise_type": noise,
                    "snr_db": snr,
                    "signal": {"missing_leads": {"mse": 0.1 + 0.01 * value}},
                }
                for noise in ("bw", "em", "ma")
                for snr in (24, 12, 6, 0)
            }
            result_models[model_id] = {"robustness": {"conditions": conditions}}
            echo_models[model_id] = {
                "factorial_mask": mask,
                "signal": {
                    "missing_leads": {
                        "r2": 0.5 + 0.01 * value,
                        "pearson": 0.7 + 0.01 * value,
                    }
                },
            }

    heatmaps(table, tmp_path)
    forest_plot(effects, tmp_path)
    pareto_plot(table, tmp_path)
    degradation_plot(
        {"models": result_models},
        {"models": registry_models},
        tmp_path,
    )
    echonext_plots({"models": echo_models}, tmp_path)

    for name in (
        "factorial_heatmaps",
        "factorial_heatmaps_4factor_supplementary",
        "factorial_effect_forest",
        "morphology_diagnostic_pareto",
        "nstdb_degradation",
        "echonext_external_heatmap",
    ):
        for suffix in (".png", ".svg", ".pdf"):
            assert (tmp_path / "plots" / f"{name}{suffix}").is_file()
