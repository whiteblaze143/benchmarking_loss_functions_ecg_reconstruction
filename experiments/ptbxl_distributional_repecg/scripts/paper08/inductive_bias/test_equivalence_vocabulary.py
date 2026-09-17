from __future__ import annotations

import numpy as np

from repecg.paper08_tokens import (
    assign_with_unknown,
    certified_clusters,
    cluster_map,
    exact_size_random_partitions,
    frequency_matched_random_partitions,
    patient_block_simultaneous_bounds,
    patient_construction_mask,
    phase_balanced_indices,
    unique_patient_support,
)


def test_patient_partition_is_deterministic_and_patient_disjoint() -> None:
    patient_ids = np.repeat(np.arange(100), 3)
    first = patient_construction_mask(patient_ids, 42)
    second = patient_construction_mask(patient_ids, 42)
    assert np.array_equal(first, second)
    for patient in np.unique(patient_ids):
        assert len(np.unique(first[patient_ids == patient])) == 1
    assert set(patient_ids[first]).isdisjoint(set(patient_ids[~first]))


def test_phase_sampling_is_balanced_and_record_specific() -> None:
    phases = phase_balanced_indices(np.arange(64), per_record=8, seed=42)
    counts = np.bincount(phases.reshape(-1), minlength=16)
    assert counts.max() - counts.min() <= 1
    assert all(len(np.unique(row)) == 8 for row in phases)
    assert len({tuple(row) for row in phases}) == 16
    assert all(np.max(np.diff(np.sort(row))) <= 3 for row in phases)


def test_patient_block_bounds_and_support_use_unique_patients() -> None:
    rng = np.random.default_rng(7)
    patient_ids = np.repeat(np.arange(80), 4)
    codes = np.tile(np.array([0, 0, 1, 1]), 80)
    patient_effect = rng.normal(scale=0.5, size=(80, 2))
    features = patient_effect[patient_ids] + (codes[:, None] * np.array([0.2, 0.0]))
    point, draws, upper = patient_block_simultaneous_bounds(
        features, codes, patient_ids, 2, bootstraps=100, alpha=0.05, seed=42
    )
    assert point.shape == (1,)
    assert draws.shape == (100, 1)
    assert upper.shape == (2, 2)
    assert upper[0, 1] == upper[1, 0]
    assert np.array_equal(unique_patient_support(codes, patient_ids, 2), [80, 80])


def test_support_gate_preserves_unresolved_codes_as_singletons() -> None:
    upper = np.array([
        [0.0, 0.1, 0.1],
        [0.1, 0.0, 0.1],
        [0.1, 0.1, 0.0],
    ])
    clusters, unresolved = certified_clusters(
        upper, delta=0.2, patient_support=np.array([50, 50, 3]), minimum_patients=40
    )
    assert clusters == [(0, 1)]
    assert np.array_equal(unresolved, [2])


def test_random_controls_match_exact_cluster_size_multiset() -> None:
    sizes = np.array([1, 2, 3, 4])
    partitions = exact_size_random_partitions(sizes, base_count=10, repeats=10, seed=42)
    assert len(partitions) == 10
    for partition in partitions:
        assert sorted(map(len, partition)) == sorted(sizes.tolist())
        flattened = [value for cluster in partition for value in cluster]
        assert sorted(flattened) == list(range(10))


def test_assignment_rejects_out_of_support_states() -> None:
    prototypes = np.array([[0.0, 0.0], [10.0, 0.0]])
    radii = np.array([1.0, 1.0])
    assigned = assign_with_unknown(
        np.array([[0.5, 0.0], [9.5, 0.0], [5.0, 5.0]]), prototypes, radii
    )
    assert np.array_equal(assigned, [0, 1, -1])
    assert np.array_equal(cluster_map([(0, 2), (1,)], 3), [0, 1, 0])


def test_frequency_matched_controls_preserve_sizes_and_improve_usage_match() -> None:
    sizes = np.array([2, 2])
    frequencies = np.array([100, 90, 10, 5])
    target = np.array([190, 15])
    partitions, scores = frequency_matched_random_partitions(
        sizes, np.arange(4), frequencies, target, repeats=5,
        candidates_per_repeat=100, seed=9,
    )
    assert len(partitions) == 5
    assert np.isfinite(scores).all()
    for partition in partitions:
        assert sorted(map(len, partition)) == [2, 2]
        assert sorted(value for cluster in partition for value in cluster) == list(range(4))
