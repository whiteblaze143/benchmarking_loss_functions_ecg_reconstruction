from __future__ import annotations

import numpy as np
import torch

from repecg.common.models import PhaseCNN, RecurrenceCNN
from repecg.paper04_hankel import hankel_descriptor
from repecg.paper05_koopman import koopman_operator, operator_descriptor, soft_observables
from repecg.paper06_conditional import conditional_distance, decompose_macro_residual, soft_membership
from repecg.paper07_operator import OperatorSetModel, operator_waveform, sample_sparse_operators
from repecg.paper08_tokens import complete_linkage_merge, odd_even_agreement, simultaneous_upper_bounds


def test_neural_shapes() -> None:
    assert PhaseCNN(128)(torch.randn(3, 16, 128)).shape == (3, 5)
    assert RecurrenceCNN()(torch.randn(3, 16, 16)).shape == (3, 5)
    model = OperatorSetModel()
    assert model(torch.randn(2, 4, 8), torch.randn(2, 4, 16, 128)).shape == (2, 5)


def test_hankel_descriptor_is_finite_and_order_sensitive() -> None:
    rng = np.random.default_rng(42)
    cell = np.cumsum(rng.normal(size=(32, 8)), axis=0)
    original = hankel_descriptor(cell)
    shuffled = hankel_descriptor(cell[rng.permutation(32)])
    assert original.shape == shuffled.shape
    assert np.isfinite(original).all()
    assert not np.allclose(original, shuffled)


def test_koopman_occupancy_preserved_by_shuffle() -> None:
    rng = np.random.default_rng(42)
    means = rng.normal(size=(64, 16))
    anchors = rng.normal(size=(32, 16))
    observable = soft_observables(means, anchors, 1.0)
    shuffled = observable[rng.permutation(len(observable))]
    assert np.allclose(observable.mean(axis=0), shuffled.mean(axis=0))
    operator = koopman_operator(observable)
    descriptor = operator_descriptor(operator, observable)
    assert operator.shape == (32, 32)
    assert np.isfinite(descriptor).all()


def test_conditional_distance_and_decomposition() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 8))
    basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    macro, residual = decompose_macro_residual(x, np.zeros(8), basis)
    assert np.allclose(x, macro @ basis.T + residual)
    anchors = rng.normal(size=(16, 3))
    membership = soft_membership(macro, anchors, 1.0)
    features = rng.normal(size=(128, 32))
    distance = conditional_distance([features] * 4, [membership] * 4)
    assert np.allclose(distance, 0)


def test_operator_measurement_law() -> None:
    rng = np.random.default_rng(42)
    basis = rng.normal(size=(500, 8))
    q = sample_sparse_operators(1, rng)[0]
    assert np.allclose(operator_waveform(basis, -q), -operator_waveform(basis, q))


def test_equivalence_tools() -> None:
    point = np.array([0.1, 0.2, 0.3])
    boot = np.stack([point - 0.01, point + 0.02, point - 0.03])
    upper = simultaneous_upper_bounds(point, boot)
    assert np.all(upper >= point)
    matrix = np.array([[0, 0.1, 0.8], [0.1, 0, 0.7], [0.8, 0.7, 0]])
    assert complete_linkage_merge(matrix, 0.2) == [(0, 1), (2,)]
    assert odd_even_agreement(np.array([5, 5]), np.array([5, 5])) == 1.0
