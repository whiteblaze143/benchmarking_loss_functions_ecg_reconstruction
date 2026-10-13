#!/usr/bin/env python3
"""
Unit Test: Rhythm Channel Extraction and Standardized Nyström RKHS Embedding.
Tests:
- 2D Poincaré atom extraction
- Coordinate standardization and deterministic median bandwidth
- Mathematical atom permutation invariance (and raw beat permutation non-invariance)
- Trivial 3-D baseline extraction (h_RR)
- R-peak detector quality audit
- Serialization and reload consistency
"""

import sys
from pathlib import Path
import numpy as np

# Ensure path
sys.path.insert(0, str(Path("experiments/ptbxl_distributional_repecg/src").resolve()))

from repecg.paper07_operator.rhythm import (
    extract_rr_atoms,
    extract_simple_rr_features,
    audit_rpeak_quality,
    RhythmResponseMap,
)


def test_regular_sinus():
    # 60 bpm: peaks every 500 samples at fs=500 (1.0s)
    rpeaks = np.arange(0, 15000, 500)
    atoms = extract_rr_atoms(rpeaks, fs=500)
    assert atoms.shape == (len(rpeaks) - 2, 2)
    # log(1.0) = 0.0, diff = 0.0
    np.testing.assert_allclose(atoms[:, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(atoms[:, 1], 0.0, atol=1e-5)
    print("  test_regular_sinus: PASS")


def test_irregular_rhythm():
    # Simulated AF: random intervals between 400ms and 1200ms
    rng = np.random.default_rng(42)
    intervals_ms = rng.uniform(400, 1200, size=30)
    rpeaks = np.cumsum(np.concatenate([[0], intervals_ms * 0.5]))  # at 500 Hz
    atoms = extract_rr_atoms(rpeaks, fs=500)
    assert atoms.shape[0] == len(rpeaks) - 2
    assert atoms.shape[1] == 2
    assert np.std(atoms[:, 0]) > 0.1  # Significant log-RR variance
    assert np.std(atoms[:, 1]) > 0.1  # Significant delta-log-RR variance
    print("  test_irregular_rhythm: PASS")


def test_atom_permutation_invariance():
    # Mathematical property: KME is a set average of atoms phi(u_i),
    # so permuting the 2-D atoms u_i leaves the mean embedding unchanged.
    rng = np.random.default_rng(999)
    atoms = rng.normal(size=(25, 2)).astype(np.float32)

    rm = RhythmResponseMap()
    rm.fit([atoms], n_landmarks=16, seed=42)

    emb_orig = rm.mean_embedding(atoms, min_intervals=5)

    # Permute atom rows
    perm_idx = rng.permutation(len(atoms))
    atoms_perm = atoms[perm_idx]
    emb_perm = rm.mean_embedding(atoms_perm, min_intervals=5)

    np.testing.assert_allclose(emb_orig, emb_perm, atol=1e-6)
    print("  test_atom_permutation_invariance: PASS")


def test_raw_peak_permutation_sensitivity():
    # Sequential local order matters for Δlog RR:
    # Permuting the underlying intervals changes the sequence of Δlog RR,
    # demonstrating that local temporal dynamics are encoded.
    rng = np.random.default_rng(123)
    intervals = np.array([0.5, 0.6, 0.7, 0.8, 1.2, 0.4, 0.9, 1.1])
    peaks1 = np.cumsum(np.concatenate([[0], intervals * 500]))
    atoms1 = extract_rr_atoms(peaks1, fs=500)

    # Permute intervals
    intervals_perm = rng.permutation(intervals)
    peaks2 = np.cumsum(np.concatenate([[0], intervals_perm * 500]))
    atoms2 = extract_rr_atoms(peaks2, fs=500)

    # The sets of atoms are different because Δlog RR depends on adjacent pairs
    assert not np.allclose(np.sort(atoms1[:, 1]), np.sort(atoms2[:, 1]))
    print("  test_raw_peak_permutation_sensitivity: PASS")


def test_simple_rr_features():
    # Test trivial 3-D baseline
    rpeaks = np.arange(0, 10000, 500)  # 1.0s intervals, 20 peaks
    h = extract_simple_rr_features(rpeaks, fs=500)
    assert h.shape == (3,)
    np.testing.assert_allclose(h[0], 0.0, atol=1e-5)  # mean(log 1.0) = 0
    np.testing.assert_allclose(h[1], 0.0, atol=1e-5)  # std(log RR) = 0
    np.testing.assert_allclose(h[2], 0.0, atol=1e-5)  # std(Δlog RR) = 0

    # Short / degenerate signal
    h_short = extract_simple_rr_features(np.array([100, 600]), fs=500)
    np.testing.assert_allclose(h_short, 0.0)
    print("  test_simple_rr_features: PASS")


def test_detector_quality_audit():
    # 10 peaks with 1 out-of-bound interval (e.g. 3.0s pause)
    intervals = [0.8, 0.8, 3.0, 0.8, 0.8, 0.8, 0.8]  # 7 intervals
    peaks = np.cumsum(np.concatenate([[0], np.array(intervals) * 500]))
    audit = audit_rpeak_quality(peaks, fs=500)
    assert audit["n_detected_peaks"] == 8
    assert audit["n_accepted_rr"] == 6  # 3.0s interval rejected
    assert audit["is_valid"] is True    # 6 >= 5
    assert 0.1 < audit["rr_rejection_fraction"] < 0.2
    print("  test_detector_quality_audit: PASS")


def test_rhythm_response_map_serialization():
    rng = np.random.default_rng(123)
    train_atoms = []
    for _ in range(30):
        intervals = rng.uniform(0.4, 1.2, size=rng.integers(10, 30))
        peaks = np.cumsum(np.concatenate([[0], intervals * 500]))
        train_atoms.append(extract_rr_atoms(peaks, fs=500))

    rm = RhythmResponseMap()
    rm.fit(train_atoms, n_landmarks=32, seed=42)
    assert rm.dim == 32
    assert rm.landmarks.shape == (32, 2)
    assert rm.whitening.shape == (32, 32)
    assert rm.mean_u.shape == (2,)
    assert rm.std_u.shape == (2,)

    test_u = train_atoms[0]
    r_rr = rm.mean_embedding(test_u)
    assert r_rr.shape == (32,)
    assert not np.isnan(r_rr).any()

    # Save and reload
    tmp_path = Path("/tmp/test_rhythm_fit.npz")
    rm.save(tmp_path)
    rm_loaded = RhythmResponseMap.load(tmp_path)
    np.testing.assert_allclose(rm.landmarks, rm_loaded.landmarks)
    np.testing.assert_allclose(rm.whitening, rm_loaded.whitening)
    np.testing.assert_allclose(rm.mean_u, rm_loaded.mean_u)
    np.testing.assert_allclose(rm.std_u, rm_loaded.std_u)
    np.testing.assert_allclose(rm.gamma, rm_loaded.gamma)
    tmp_path.unlink()
    print("  test_rhythm_response_map_serialization: PASS")


def main():
    print("Running Rhythm Channel Unit Tests...")
    test_regular_sinus()
    test_irregular_rhythm()
    test_atom_permutation_invariance()
    test_raw_peak_permutation_sensitivity()
    test_simple_rr_features()
    test_detector_quality_audit()
    test_rhythm_response_map_serialization()
    print("ALL RHYTHM UNIT TESTS PASSED!")


if __name__ == "__main__":
    main()
