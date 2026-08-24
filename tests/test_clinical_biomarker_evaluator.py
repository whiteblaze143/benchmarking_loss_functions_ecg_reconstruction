import numpy as np

import scripts.evaluate_clinical_biomarkers_multids as evaluator


def _fake_lead_result(value, n_samples=32):
    mask = np.zeros(n_samples, dtype=bool)
    mask[4:9] = True
    boundaries = np.array([4.0, 20.0])
    offsets = np.array([8.0, 24.0])
    return {
        "qrs_ms": float(value),
        "st_mv": float(value),
        "rpeaks": np.array([6, 22]),
        "p_onsets": boundaries,
        "p_offsets": offsets,
        "r_onsets": boundaries,
        "r_offsets": offsets,
        "t_onsets": boundaries,
        "t_offsets": offsets,
        "p_mask": mask.copy(),
        "qrs_mask": mask.copy(),
        "t_mask": mask.copy(),
    }


def test_missing_lead_contract_excludes_all_three_observed_leads():
    assert evaluator.OBSERVED_LEAD_INDICES == (0, 1, 7)
    assert evaluator.MISSING_LEAD_INDICES == (2, 3, 4, 5, 6, 8, 9, 10, 11)
    assert evaluator.MISSING_LEAD_NAMES == (
        "III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6"
    )


def test_delineation_visits_all_missing_leads_independently_and_never_v2(monkeypatch):
    visited = []

    def fake_extract(lead, fs=500):
        lead_index = int(lead[0])
        visited.append(lead_index)
        return _fake_lead_result(lead_index, n_samples=len(lead))

    monkeypatch.setattr(evaluator, "_extract_lead_biomarkers", fake_extract)
    signal = np.stack([np.full(32, index, dtype=float) for index in range(12)])
    result = evaluator._extract_missing_lead_biomarkers(signal)

    assert visited == list(evaluator.MISSING_LEAD_INDICES)
    assert 7 not in visited
    assert result["delineated_missing_leads"] == 9
    assert result["missing_lead_coverage"] == 1.0
    assert result["delineation_mode"] == "independent_missing_leads"
    assert result["qrs_ms"] == np.mean(evaluator.MISSING_LEAD_INDICES)


def test_signal_missing_summary_cannot_be_improved_by_v2_passthrough(monkeypatch):
    lead_results = {
        name: _fake_lead_result(index)
        for index, name in zip(evaluator.MISSING_LEAD_INDICES, evaluator.MISSING_LEAD_NAMES)
    }
    fake_summary = {
        "qrs_ms": 100.0,
        "lvh_mv": 2.0,
        "st_lead_mv": {name: 0.0 for name in evaluator.MISSING_LEAD_NAMES},
        "lead_results": lead_results,
        "delineated_missing_leads": 9,
        "missing_lead_coverage": 1.0,
    }
    monkeypatch.setattr(
        evaluator,
        "_extract_missing_lead_biomarkers",
        lambda signal, fs=500: fake_summary,
    )

    rng = np.random.default_rng(7)
    target = rng.normal(size=(12, 32))
    reconstruction = target.copy()
    reconstruction[7] += 1000.0  # Only observed V2 is deliberately wrong.
    result = evaluator.extract_biomarkers_kernel((target, reconstruction))

    assert result is not None
    missing_summary = result[3]
    assert missing_summary["val_missing_mse"] == 0.0
    assert missing_summary["val_missing_pearson"] == 1.0


def test_patient_bootstrap_is_paired_and_clustered():
    labels = np.array([0, 0, 1, 1, 0, 1])
    reference = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    reconstructed = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.6])
    patients = np.array([10, 10, 20, 20, 30, 30])

    result = evaluator.patient_cluster_bootstrap_delta(
        labels,
        reference,
        reconstructed,
        patients,
        "brier",
        n_bootstraps=50,
    )

    assert result["n_records"] == 6
    assert result["n_patients"] == 3
    assert result["n_bootstraps"] == 50
    assert np.isclose(
        result["delta"],
        np.mean((reconstructed - labels) ** 2) - np.mean((reference - labels) ** 2),
    )
