from pathlib import Path

import numpy as np

from theta_repspat.panobench import PANOBENCH_ANGLES_RAD, ReleasedPanoBench


def test_downloaded_panobench_adapter_is_deterministic():
    root = Path("/data/mithunmanivannan/panobench/train")
    dataset = ReleasedPanoBench(root, seed=123)
    first = dataset[0]
    repeated = dataset[0]

    assert len(dataset) == 3440
    assert first["input"].shape == (3, 4608)
    assert first["target"].shape == (1, 4608)
    assert PANOBENCH_ANGLES_RAD.shape == (44, 2)
    np.testing.assert_array_equal(first["input"], repeated["input"])
    np.testing.assert_array_equal(first["target"], repeated["target"])
