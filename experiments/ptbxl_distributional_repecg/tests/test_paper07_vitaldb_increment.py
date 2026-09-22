from __future__ import annotations

import numpy as np

from scripts.paper07.run_vitaldb_increment_probe import (
    IMQNystromMap,
    _features,
    _split_indices,
    _subject_split,
    _wrong_subject_permutation,
)


def test_subject_hash_split_is_stable_and_subject_disjoint() -> None:
    subjects = np.asarray([f"p{index // 2}" for index in range(400)])
    split = _split_indices(subjects, seed=42)
    sets = {name: set(subjects[index]) for name, index in split.items()}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])
    assert _subject_split("p17", 42) == _subject_split("p17", 42)


def test_wrong_subject_permutation_is_exact_derangement() -> None:
    subjects = np.asarray(["a", "a", "b", "c", "d", "e", "f", "g"])
    permutation = _wrong_subject_permutation(subjects, seed=7)
    assert sorted(permutation.tolist()) == list(range(len(subjects)))
    assert np.all(subjects[permutation] != subjects)


def test_feature_arms_and_imq_map_contract() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(size=(30, 3, 16)).astype(np.float32)
    assert _features(values, "ii").shape == (30, 16)
    assert _features(values, "ppg").shape == (30, 16)
    assert _features(values, "ii_ppg").shape == (30, 32)
    mapping = IMQNystromMap.fit(_features(values, "ii"), landmarks=8, c2=1.0, seed=9)
    transformed = mapping.transform(_features(values, "ii"))
    assert transformed.shape == (30, 8)
    assert np.isfinite(transformed).all()
