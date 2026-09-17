import numpy as np

from repecg.paper04_hankel import DESCRIPTOR_DIMENSION, hankel_descriptor, transform_cell


def _cell() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 32)
    return np.stack([np.sin((axis + 1) * t) + axis * t for axis in range(8)], axis=1)


def test_descriptor_has_frozen_finite_dimension() -> None:
    descriptor = hankel_descriptor(_cell())
    assert descriptor.shape == (DESCRIPTOR_DIMENSION,)
    assert DESCRIPTOR_DIMENSION == 30
    assert np.isfinite(descriptor).all()


def test_time_destroyer_preserves_endpoints_and_samples() -> None:
    cell = _cell()
    destroyed = transform_cell(
        cell, kind="time_shuffle", record_id=11, beat=2, phase=3, seed=42
    )
    assert np.array_equal(destroyed[[0, -1]], cell[[0, -1]])
    assert {tuple(row) for row in destroyed} == {tuple(row) for row in cell}
    assert not np.allclose(hankel_descriptor(destroyed), hankel_descriptor(cell))


def test_warp_sham_is_deterministic_monotone_and_endpoint_preserving() -> None:
    cell = _cell()
    sham = transform_cell(
        cell, kind="monotone_warp_sham", record_id=11, beat=2, phase=3, seed=42
    )
    assert np.allclose(sham[[0, -1]], cell[[0, -1]])
    assert np.array_equal(
        sham,
        transform_cell(
            cell, kind="monotone_warp_sham", record_id=11, beat=2, phase=3, seed=42
        ),
    )
