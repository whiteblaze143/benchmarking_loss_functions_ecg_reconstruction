"""
Unit Tests for prototype/metrics_engine.py
"""
import pytest
import numpy as np
import pandas as pd
from prototype.metrics_engine import (
    extract_continuous_episodes,
    match_episodes,
    compute_clinical_false_alerts_per_24h,
    compute_burden_metrics,
    patient_clustered_bootstrap_ci,
)

def test_extract_continuous_episodes():
    # 10 windows: 0, 1, 1, 1, 0, 0, 1, 0, 1, 1 (window_sec=30s)
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    # Episode 1: idx 1 to 4 -> duration = 3 * 30 = 90s >= 30s
    # Episode 2: idx 6 to 7 -> duration = 1 * 30 = 30s >= 30s
    # Episode 3: idx 8 to 10 -> duration = 2 * 30 = 60s >= 30s
    eps = extract_continuous_episodes(mask, window_sec=30.0, min_duration_sec=30.0)
    assert len(eps) == 3
    assert eps[0]["start_sec"] == 30.0
    assert eps[0]["duration_sec"] == 90.0
    assert eps[1]["duration_sec"] == 30.0
    assert eps[2]["duration_sec"] == 60.0

    # With min_duration_sec = 60s, Episode 2 should be excluded
    eps60 = extract_continuous_episodes(mask, window_sec=30.0, min_duration_sec=60.0)
    assert len(eps60) == 2

def test_match_episodes():
    ref = [
        {"start_sec": 100.0, "end_sec": 200.0, "duration_sec": 100.0},
        {"start_sec": 500.0, "end_sec": 800.0, "duration_sec": 300.0},
    ]
    pred = [
        {"start_sec": 110.0, "end_sec": 190.0, "duration_sec": 80.0},  # matches ref[0]
        {"start_sec": 250.0, "end_sec": 300.0, "duration_sec": 50.0},  # false alarm
    ]
    res = match_episodes(ref, pred, min_overlap_sec=15.0)
    assert res["n_ref"] == 2
    assert res["detected_ref"] == 1
    assert res["false_pred"] == 1
    assert res["sensitivity"] == 0.5

def test_compute_clinical_false_alerts_per_24h():
    # 24 hours = 2880 windows (30s each)
    mask = np.zeros(2880, dtype=bool)
    # 5 false alarms within 10 minutes (indices 10 to 30) -> should be merged into 1 alert
    mask[10:15] = True
    # 1 false alarm at hour 12 (index 1440) -> 2nd alert
    mask[1440] = True
    
    res = compute_clinical_false_alerts_per_24h(mask, monitored_hours=24.0, window_sec=30.0, refractory_period_sec=3600.0)
    assert res["raw_fa"] == 6
    assert res["raw_fa_per_24h"] == 6.0
    assert res["clinical_alerts"] == 2
    assert res["alerts_per_24h"] == 2.0

def test_compute_burden_metrics():
    gt = np.array([10.0, 20.0, 50.0, 80.0])
    pred = np.array([12.0, 18.0, 55.0, 75.0])
    res = compute_burden_metrics(gt, pred)
    assert res["mae"] == 3.5
    assert res["pearson_r"] > 0.95
    assert res["spearman_rho"] == 1.0

def test_patient_clustered_bootstrap_ci():
    df = pd.DataFrame({
        "patient": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
        "error": [1.0, 2.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
    })
    def calc_mean(d):
        return {"mean_err": float(d["error"].mean())}
    ci = patient_clustered_bootstrap_ci(df, "patient", calc_mean, n_bootstraps=100, seed=42)
    assert "mean_err" in ci
    assert ci["mean_err"]["ci_lower"] <= ci["mean_err"]["point_estimate"] <= ci["mean_err"]["ci_upper"]
