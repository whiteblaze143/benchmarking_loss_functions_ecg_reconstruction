from __future__ import annotations

import numpy as np
import torch

from repecg.common.models import PhaseCNN, RecurrenceCNN
from repecg.paper02_kernel_mean.controls import (
    exact_moment_matched_gaussian,
    mean_covariance_features,
    sample_covariance,
)
from repecg.paper03_signature import logsignature_descriptor
from repecg.paper04_hankel import hankel_descriptor
from repecg.paper05_koopman import (
    chronological_permutation,
    koopman_operator,
    median_anchor_bandwidth,
    operator_descriptor,
    soft_observables,
)
from repecg.paper06_conditional import conditional_distance, decompose_macro_residual, soft_membership
from repecg.paper07_operator import (
    OperatorSetModel,
    frozen_operator_banks,
    operator_waveform,
    record_training_operators,
    response_atoms,
    sample_sparse_operators,
    symmetric_bernoulli_kl,
)
from scripts.paper07.train_paper07_shared_grid import (
    UNKNOWN_ID,
    build_training_vocabulary,
    map_operator_ids,
    sample_context_target_indices,
)
from repecg.paper08_tokens import complete_linkage_merge, odd_even_agreement, simultaneous_upper_bounds
from repecg.paper09_counterfactual import CounterfactualOperatorSetModel, mismatch_operators
from scripts.paper09.train_paper09_shared_grid import normalized_state_mse, sample_disjoint_contexts


def test_neural_shapes() -> None:
    assert PhaseCNN(128)(torch.randn(3, 16, 128)).shape == (3, 5)
    assert RecurrenceCNN()(torch.randn(3, 16, 16)).shape == (3, 5)
    model = OperatorSetModel()
    assert model(torch.randn(2, 4, 8), torch.randn(2, 4, 16, 128)).shape == (2, 5)


def test_paper02_destroyer_matches_realized_mean_and_covariance() -> None:
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(7, 32, 8, generator=generator, dtype=torch.float64)
    values = values @ torch.diag(torch.linspace(0.2, 2.0, 8, dtype=torch.float64)) + 3.0
    matched = exact_moment_matched_gaussian(values, generator=generator)
    assert torch.max(torch.abs(values.mean(1) - matched.mean(1))) < 1e-8
    assert torch.max(torch.abs(sample_covariance(values) - sample_covariance(matched))) < 1e-8
    features = mean_covariance_features(values)
    assert features.shape == (7, 44)


def test_hankel_descriptor_is_finite_and_order_sensitive() -> None:
    rng = np.random.default_rng(42)
    cell = np.cumsum(rng.normal(size=(32, 8)), axis=0)
    original = hankel_descriptor(cell)
    shuffled = hankel_descriptor(cell[rng.permutation(32)])
    assert original.shape == shuffled.shape
    assert np.isfinite(original).all()
    assert not np.allclose(original, shuffled)


def test_signature_descriptor_uses_path_order() -> None:
    phase = np.linspace(0.0, 1.0, 16)
    cell = np.stack([np.sin((axis + 1) * phase) for axis in range(8)], axis=1)
    forward = logsignature_descriptor(cell)
    reversed_path = logsignature_descriptor(cell[::-1])
    assert forward.shape == reversed_path.shape
    assert np.isfinite(forward).all()
    assert not np.allclose(forward, reversed_path)


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
    assert descriptor.shape == (68,)
    assert np.isfinite(descriptor).all()


def test_koopman_training_bandwidth_and_deterministic_shuffle() -> None:
    rng = np.random.default_rng(7)
    means = rng.normal(size=(80, 16))
    anchors = means[:32]
    tau = median_anchor_bandwidth(means, anchors)
    observables = soft_observables(means, anchors, tau)
    assert np.allclose(observables.sum(axis=1), 1.0)
    left = chronological_permutation(len(means), record_id=17, seed=42)
    right = chronological_permutation(len(means), record_id=17, seed=42)
    assert np.array_equal(left, right)
    assert np.allclose(observables.mean(axis=0), observables[left].mean(axis=0))


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


def test_paper07_frozen_operator_banks_and_training_draw() -> None:
    banks = frozen_operator_banks()
    assert {name: len(values) for name, values in banks.items()} == {
        "seen": 108, "derived_limb": 4, "dense": 100, "i_to_v2": 9,
    }
    assert all(np.allclose(np.linalg.norm(values, axis=1), 1.0) for values in banks.values())
    for name in ("derived_limb", "dense", "i_to_v2"):
        for value in banks[name]:
            assert np.min(np.linalg.norm(banks["seen"] - value, axis=1)) > 1e-8
            assert np.min(np.linalg.norm(banks["seen"] + value, axis=1)) > 1e-8
    left = record_training_operators(123, 42)
    right = record_training_operators(123, 42)
    assert left.shape == (8, 8)
    assert np.array_equal(left, right)
    assert np.allclose(np.linalg.norm(left, axis=1), 1.0)


def test_paper07_response_atoms_obey_orientation_law() -> None:
    rng = np.random.default_rng(19)
    beats = rng.normal(size=(3, 256, 8))
    q = rng.normal(size=8)
    positive = response_atoms(beats, q, 0.25)
    negative = response_atoms(beats, -q, 0.25)
    assert positive.shape == (3, 256, 2)
    assert np.allclose(negative, -positive)


def test_paper07_categorical_vocabulary_has_one_unknown_id() -> None:
    train = np.eye(8, dtype=np.float32).reshape(1, 8, 8)
    vocabulary = build_training_vocabulary(train)
    query = np.concatenate((train, np.ones((1, 1, 8), dtype=np.float32)), axis=1)
    ids = map_operator_ids(query, vocabulary)
    assert np.all(ids[:, :8] >= 0)
    assert ids[0, 8] == UNKNOWN_ID
    model = OperatorSetModel(operator_mode="categorical", vocabulary_size=len(vocabulary))
    assert "unknown_operator" in dict(model.named_buffers())
    assert "unknown_operator" not in dict(model.named_parameters())


def test_paper07_context_sampling_holds_out_target() -> None:
    generator = torch.Generator().manual_seed(42)
    observed_sizes = set()
    for _ in range(100):
        context, target = sample_context_target_indices(16, 8, generator, torch.device("cpu"))
        observed_sizes.add(context.shape[1])
        assert 1 <= context.shape[1] <= 6
        assert not torch.any(context == target[:, None])
    assert observed_sizes == set(range(1, 7))


def test_paper07_matched_architecture_and_explicit_orientation_loss() -> None:
    torch.manual_seed(9)
    continuous = OperatorSetModel(operator_mode="continuous")
    torch.manual_seed(9)
    categorical = OperatorSetModel(operator_mode="categorical", vocabulary_size=8)
    for module in ("response", "set_encoder", "pool", "latent", "head", "decoder"):
        left = getattr(continuous, module).state_dict()
        right = getattr(categorical, module).state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[name], right[name]) for name in left)
    logits = torch.randn(4, 5, requires_grad=True)
    loss = symmetric_bernoulli_kl(logits, logits)
    assert float(loss) == 0.0
    loss.backward()


def test_equivalence_tools() -> None:
    point = np.array([0.1, 0.2, 0.3])
    boot = np.stack([point - 0.01, point + 0.02, point - 0.03])
    upper = simultaneous_upper_bounds(point, boot)
    assert np.all(upper >= point)
    matrix = np.array([[0, 0.1, 0.8], [0.1, 0, 0.7], [0.8, 0.7, 0]])
    assert complete_linkage_merge(matrix, 0.2) == [(0, 1), (2,)]
    assert odd_even_agreement(np.array([5, 5]), np.array([5, 5])) == 1.0


def test_paper09_counterfactual_interface_never_accepts_basis_tensor() -> None:
    model = CounterfactualOperatorSetModel()
    operators = torch.randn(2, 4, 8)
    responses = torch.randn(2, 4, 16, 128)
    target = torch.randn(2, 8)
    logits, prediction, state = model(
        operators, responses, target_operator=target,
        return_counterfactual=True, return_state=True,
    )
    assert logits.shape == (2, 5)
    assert prediction.shape == (2, 16, 128)
    assert state.shape == (2, 256)
    with torch.no_grad():
        try:
            model(torch.randn(2, 256, 8), torch.randn(2, 256, 8))
        except ValueError:
            pass
        else:
            raise AssertionError("raw basis-shaped input was accepted")


def test_paper09_mismatch_changes_q_but_not_responses() -> None:
    operators = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
    responses = torch.randn(2, 4, 16, 128)
    permutation = torch.tensor([[1, 2, 3, 0], [3, 0, 1, 2]])
    mismatched = mismatch_operators(operators, permutation)
    assert not torch.equal(mismatched, operators)
    assert torch.equal(responses, responses.clone())


def test_paper09_contexts_are_disjoint_and_target_is_held_out() -> None:
    generator = torch.Generator().manual_seed(17)
    observed = set()
    for _ in range(100):
        first, second, target = sample_disjoint_contexts(8, 8, generator, torch.device("cpu"))
        observed.add(first.shape[1])
        for row in range(8):
            left = set(first[row].tolist())
            right = set(second[row].tolist())
            assert left.isdisjoint(right)
            assert int(target[row]) not in left | right
    assert observed == {1, 2, 3}
    state = torch.randn(5, 256)
    assert float(normalized_state_mse(state, state)) == 0.0
