from __future__ import annotations

import numpy as np

from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.paper10_interventional.paired_views import (
    Environment,
    apply_environment,
    environment_bank,
    phase_kme,
    view_seed,
)


def test_frozen_environment_bank_has_primary_and_boundary_tiers():
    bank = environment_bank()
    assert [entry.name for entry in bank[:5]] == [
        "clean", "gain_0p8", "gain_1p2", "resample_500_250_500", "noise_20db_r0",
    ]
    assert sum(entry.tier == "primary" for entry in bank) == 7
    assert sum(entry.tier == "boundary" for entry in bank) == 3


def test_noise_views_are_reproducible_and_realization_specific():
    signal = np.random.default_rng(1).normal(size=(5000, 12))
    first = Environment("noise_20db_r0", "noise", 20.0, 0)
    second = Environment("noise_20db_r1", "noise", 20.0, 1)
    kwargs = {"patient_id": 11, "ecg_id": 22, "master_seed": 42}
    a, seed_a = apply_environment(signal, first, **kwargs)
    b, seed_b = apply_environment(signal, first, **kwargs)
    c, seed_c = apply_environment(signal, second, **kwargs)
    assert seed_a == seed_b == view_seed(environment=first, **kwargs)
    assert seed_a != seed_c
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_raw_interventions_preserve_required_shape():
    signal = np.random.default_rng(2).normal(size=(5000, 12))
    common = {"patient_id": 1, "ecg_id": 2, "master_seed": 3}
    for environment in environment_bank():
        view, _ = apply_environment(signal, environment, **common)
        assert view.shape == signal.shape
    gain, _ = apply_environment(signal, Environment("gain", "gain", 0.8), **common)
    assert np.allclose(gain, 0.8 * signal)


def test_phase_kme_uses_a_frozen_map_without_refitting():
    rng = np.random.default_rng(3)
    fit = rng.normal(size=(128, 8))
    whitening = WhiteningTransform.fit(fit)
    mapping = NystromMap.fit(whitening.transform(fit), landmarks=8, seed=3)
    result = phase_kme(rng.normal(size=(3, 256, 8)), whitening, mapping)
    assert result.shape == (16, 8)
    assert np.isfinite(result).all()
