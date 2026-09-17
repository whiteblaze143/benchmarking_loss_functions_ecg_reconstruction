from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _trainer_module():
    path = Path(__file__).parents[1] / "train_paper08_shared_grid.py"
    spec = importlib.util.spec_from_file_location("paper08_factorial_trainer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factorial_representations_preserve_unknown_control_semantics() -> None:
    trainer = _trainer_module()
    payload = {
        "continuous": np.zeros((2, 16, 3), dtype=np.float32),
        "fine_kmeans_ids": np.tile(np.array([0, 1], dtype=np.int64), (2, 8)),
        "size_kmeans_ids": np.tile(np.array([0, 1], dtype=np.int64), (2, 8)),
        "equivalence_token_ids": np.tile(np.array([0, -1], dtype=np.int64), (2, 8)),
    }
    certificate = {
        "token_prototypes": np.zeros((2, 3), dtype=np.float32),
        "random_merge_maps": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "frequency_matched_random_merge_maps": np.array([[1, 0], [0, 1]], dtype=np.int64),
    }
    unknown_only, _ = trainer._representation_features(
        payload, certificate, "unk_pattern_only", 0
    )
    hidden_unknown, _ = trainer._representation_features(
        payload, certificate, "token_without_unk_signal", 0
    )
    equivalence, _ = trainer._representation_features(payload, certificate, "equivalence", 0)
    random_zero, design_zero = trainer._representation_features(
        payload, certificate, "random_merge", 0
    )
    random_one, design_one = trainer._representation_features(
        payload, certificate, "random_merge", 1
    )

    assert unknown_only.shape == (2, 16, 1)
    assert np.array_equal(unknown_only[..., 0], payload["equivalence_token_ids"] < 0)
    assert np.all(hidden_unknown[payload["equivalence_token_ids"] < 0] == 0)
    assert np.all(equivalence[payload["equivalence_token_ids"] < 0, -1] == 1)
    assert design_zero["random_control"] == 0
    assert design_one["random_control"] == 1
    assert not np.array_equal(random_zero, random_one)
