import numpy as np

from scripts.paper01.build_paper01_representations import (
    _allowed_mask, fit_tau, intervene, recurrence,
)


def test_recurrence_contract_and_train_fitted_scale() -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(7, 16, 5))
    tau = fit_tau(train)
    operator = recurrence(train, tau, batch_size=3)
    assert tau > 0
    assert operator.shape == (7, 16, 16)
    assert np.isfinite(operator).all()
    assert np.allclose(operator, operator.transpose(0, 2, 1), atol=1e-6)
    assert np.all(operator[:, ~_allowed_mask()] == 0)


def test_recurrence_rejects_wrong_phase_count() -> None:
    values = np.zeros((2, 15, 3))
    try:
        fit_tau(values)
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong phase count was accepted")


def test_record_keyed_destroyer_and_sham_are_deterministic() -> None:
    values = np.arange(2 * 16 * 3).reshape(2, 16, 3)
    ecg_ids = np.array([101, 202])
    destroyed = intervene(values, ecg_ids, kind="phase_content_permutation", seed=42)
    sham = intervene(values, ecg_ids, kind="cyclic_relabel_sham", seed=42)
    assert np.array_equal(
        destroyed,
        intervene(values, ecg_ids, kind="phase_content_permutation", seed=42),
    )
    assert not np.array_equal(destroyed, values)
    for original, shifted in zip(values, sham):
        assert {tuple(row) for row in original} == {tuple(row) for row in shifted}
        assert any(np.array_equal(shifted, np.roll(original, shift, axis=0)) for shift in range(1, 16))
