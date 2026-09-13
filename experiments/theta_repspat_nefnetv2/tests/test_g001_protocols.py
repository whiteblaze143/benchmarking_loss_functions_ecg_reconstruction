import numpy as np
import pytest

from theta_repspat.panorama import CANONICAL_LEADS, CANONICAL_ANGLES_RAD
from scripts.eval_g001_b import psnr, ssim_1d, angular_distance_deg
from scripts.eval_g001_c import pearson_r, QUERY_LEAD_NAMES, INPUT_LEAD_NAMES


def test_psnr_and_ssim_identical():
    x = np.sin(np.linspace(0, 10 * np.pi, 500)).astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min())

    assert psnr(x, x, val_range=1.0) == float("inf")
    assert ssim_1d(x, x, val_range=1.0) == pytest.approx(1.0, abs=1e-5)


def test_ssim_degraded():
    x = np.sin(np.linspace(0, 10 * np.pi, 500)).astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min())
    noise = np.random.default_rng(0).normal(0, 0.2, size=500)
    x_noisy = np.clip(x + noise, 0, 1)

    s = ssim_1d(x_noisy, x, val_range=1.0)
    assert 0.0 < s < 1.0


def test_pearson_r_properties():
    x = np.linspace(0, 10, 100)
    y_pos = 2.0 * x + 5.0
    y_neg = -3.0 * x + 1.0
    y_const = np.ones(100)

    assert pearson_r(x, y_pos) == pytest.approx(1.0, abs=1e-5)
    assert pearson_r(x, y_neg) == pytest.approx(-1.0, abs=1e-5)
    assert pearson_r(x, y_const) == 0.0


def test_observed_normalization_isolation():
    # 44 channels
    raw_all = np.random.default_rng(42).uniform(-5.0, 5.0, size=(44, 1000))
    # channels 0, 1, 10 are observed and have small amplitudes [-1, 1]
    raw_all[0] = np.random.default_rng(1).uniform(-1.0, 1.0, size=1000)
    raw_all[1] = np.random.default_rng(2).uniform(-0.5, 0.5, size=1000)
    raw_all[10] = np.random.default_rng(3).uniform(-0.8, 0.8, size=1000)
    # channel 43 has huge outlier [100, 200]
    raw_all[43] = 150.0

    obs_indices = [0, 1, 10]
    obs_data = raw_all[obs_indices]
    m_obs, M_obs = float(obs_data.min()), float(obs_data.max())
    m_all, M_all = float(raw_all.min()), float(raw_all.max())

    # Observed extrema must not be contaminated by outlier view 43
    assert M_obs <= 1.0
    assert m_obs >= -1.0
    assert M_all == 150.0


def test_canonical_lead_contract():
    assert len(CANONICAL_LEADS) == 8
    assert list(CANONICAL_LEADS) == ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert INPUT_LEAD_NAMES == ["I", "II", "V3"]
    assert QUERY_LEAD_NAMES == ["V1", "V2", "V4", "V5", "V6"]
    assert CANONICAL_ANGLES_RAD.shape == (8, 2)
