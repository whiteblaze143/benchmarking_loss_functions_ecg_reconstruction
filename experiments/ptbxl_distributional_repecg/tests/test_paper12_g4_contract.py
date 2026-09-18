from __future__ import annotations

import numpy as np
import pytest
import torch

from repecg.paper12_innovations import (
    PRIMARY_REPRESENTATIONS,
    ProbeScaler,
    minibatch_order,
    patient_bootstrap_multiplicity,
    probe_features,
    record_weights,
)


def test_primary_registry_forbids_label_matched_wrong_history():
    assert PRIMARY_REPRESENTATIONS == ("state", "phase_standardized", "mean_residual", "innovation")
    assert "wrong_label" not in PRIMARY_REPRESENTATIONS


def test_probe_features_exclude_phase_zero_and_have_frozen_width():
    values = torch.randn(3, 16, 1024)
    features = probe_features(values)
    assert features.shape == (3, 15360)
    changed = values.clone()
    changed[:, 0] += 1000
    assert torch.equal(features, probe_features(changed))


def test_probe_scaler_is_fitted_only_from_supplied_training_tensor():
    train = torch.randn(20, 15360)
    scaler = ProbeScaler.fit(train)
    held_out = torch.randn(4, 15360) + 100
    _ = scaler.transform(held_out)
    assert torch.equal(scaler.mean, train.double().mean(0, keepdim=True))


def test_minibatch_order_is_shared_by_seed_and_epoch():
    first = minibatch_order(100, 42, 3, torch.device("cpu"))
    second = minibatch_order(100, 42, 3, torch.device("cpu"))
    assert torch.equal(first, second)


def test_patient_bootstrap_is_patient_equal_and_reusable():
    patient_ids = np.array([1, 1, 2, 3, 3, 3])
    draws = patient_bootstrap_multiplicity(patient_ids, seed=12, draws=5)
    assert np.array_equal(draws, patient_bootstrap_multiplicity(patient_ids, seed=12, draws=5))
    weights = record_weights(patient_ids, np.array([1, 2, 1]))
    assert weights[patient_ids == 1].sum() == pytest.approx(1)
    assert weights[patient_ids == 2].sum() == pytest.approx(2)
    assert weights[patient_ids == 3].sum() == pytest.approx(1)
