from __future__ import annotations

import numpy as np

from scripts.paper07.run_frozen_ptbxl_conditional_probe import _derive_standard


def test_exact_lead_algebra_and_observed_passthrough() -> None:
    independent = np.zeros((2, 8, 7), dtype=np.float32)
    independent[:, 0] = 2.0
    independent[:, 1] = 5.0
    observed = np.full((2, 7), 11.0, dtype=np.float32)
    standard = _derive_standard(independent, observed_index=0, observed=observed)
    assert np.all(standard[:, 0] == 11.0)
    assert np.all(standard[:, 1] == 5.0)
    assert np.all(standard[:, 2] == -6.0)
    assert np.all(standard[:, 3] == -8.0)
    assert np.all(standard[:, 4] == 8.5)
    assert np.all(standard[:, 5] == -0.5)
