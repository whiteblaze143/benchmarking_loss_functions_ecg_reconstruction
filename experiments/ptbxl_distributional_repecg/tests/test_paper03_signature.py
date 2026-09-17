import numpy as np

from repecg.paper03_signature.representations import (
    DESCRIPTOR_DIMENSION,
    path_descriptor,
    transform_path,
)


def _path() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 16)
    return np.stack([np.sin((index + 1) * t) + index * t for index in range(8)], axis=1)


def test_depth3_descriptor_contract() -> None:
    path = _path()
    descriptor = path_descriptor(path)
    assert descriptor.shape == (DESCRIPTOR_DIMENSION,)
    assert DESCRIPTOR_DIMENSION == 228
    assert np.array_equal(descriptor[:8], path[0])
    assert np.array_equal(descriptor[8:16], path[-1])
    assert np.allclose(descriptor[16:24], path.mean(axis=0))


def test_order_destroyer_preserves_endpoints_and_sample_marginal() -> None:
    path = _path()
    destroyed = transform_path(
        path, kind="order_destroy", record_id=7, beat=2, phase=4, seed=42
    )
    assert np.array_equal(destroyed[0], path[0])
    assert np.array_equal(destroyed[-1], path[-1])
    assert {tuple(row) for row in destroyed} == {tuple(row) for row in path}
    assert np.array_equal(
        destroyed,
        transform_path(path, kind="order_destroy", record_id=7, beat=2, phase=4, seed=42),
    )


def test_monotone_warp_is_endpoint_preserving_and_weaker_than_destruction() -> None:
    path = _path()
    sham = transform_path(
        path, kind="monotone_warp_sham", record_id=7, beat=2, phase=4, seed=42
    )
    destroyed = transform_path(
        path, kind="order_destroy", record_id=7, beat=2, phase=4, seed=42
    )
    assert np.allclose(sham[[0, -1]], path[[0, -1]])
    base_descriptor = path_descriptor(path)
    assert np.linalg.norm(path_descriptor(sham) - base_descriptor) < np.linalg.norm(
        path_descriptor(destroyed) - base_descriptor
    )


def test_time_reversal_swaps_endpoints_and_changes_descriptor() -> None:
    path = _path()
    reverse = transform_path(
        path, kind="time_reverse", record_id=7, beat=2, phase=4, seed=42
    )
    assert np.array_equal(reverse[0], path[-1])
    assert np.array_equal(reverse[-1], path[0])
    assert not np.allclose(path_descriptor(reverse), path_descriptor(path))
