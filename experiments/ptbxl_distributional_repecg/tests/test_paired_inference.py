import numpy as np
import pytest

from repecg.evaluation.paired_inference import (
    paired_delong_auc,
    patient_equal_paired_bootstrap,
)


def test_paired_delong_identical_scores_has_zero_delta_and_unit_p() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])
    result = paired_delong_auc(y, scores, scores)
    assert result.auc_a == pytest.approx(1.0)
    assert result.auc_b == pytest.approx(1.0)
    assert result.delta_auc == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_paired_delong_preserves_score_pairing() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    strong = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    weak = np.array([0.40, 0.80, 0.20, 0.30, 0.90, 0.60])
    result = paired_delong_auc(y, strong, weak)
    assert result.auc_a == pytest.approx(1.0)
    assert result.auc_b == pytest.approx(6.0 / 9.0)
    assert result.delta_auc > 0.0
    assert 0.0 <= result.p_value <= 1.0


@pytest.mark.parametrize(
    ("y", "left", "right", "message"),
    [
        (np.array([0, 1]), np.array([0.1]), np.array([0.2]), "identical length"),
        (np.array([0, 2]), np.array([0.1, 0.2]), np.array([0.2, 0.3]), "binary"),
        (np.array([1, 1]), np.array([0.1, 0.2]), np.array([0.2, 0.3]), "both outcome"),
        (np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]), np.array([0.2, 0.3, 0.8]), "at least two"),
    ],
)
def test_paired_delong_rejects_invalid_contracts(y, left, right, message) -> None:
    with pytest.raises(ValueError, match=message):
        paired_delong_auc(y, left, right)


def test_patient_equal_paired_bootstrap_is_deterministic_and_paired() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    left = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    right = np.array([0.40, 0.80, 0.20, 0.30, 0.90, 0.60])
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    one = patient_equal_paired_bootstrap(y, left, right, ids, replicates=100, seed=7)
    two = patient_equal_paired_bootstrap(y, left, right, ids, replicates=100, seed=7)
    assert one == two
    assert one.observed_delta == pytest.approx(1.0 / 3.0)
    assert one.valid_replicates <= one.replicates


def test_patient_equal_paired_bootstrap_rejects_repeated_patient_rows() -> None:
    with pytest.raises(ValueError, match="exactly one row per patient"):
        patient_equal_paired_bootstrap(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.2, 0.8]), np.array(["same", "same"])
        )
