"""Unit and algebraic tests for Milestone R026 Canonical-Frame SetOperator."""

import numpy as np
import pytest

from repecg.paper07_operator.canonical_frame import (
    observable_projector,
    canonical_frame,
    canonicalize_measurements,
    canonicalize_beats,
)


def get_standard_operators():
    e = np.eye(8, dtype=np.float64)
    I = e[0]
    II = e[1]
    III = II - I
    aVR = -0.5 * (I + II)
    aVL = I - 0.5 * II
    aVF = II - 0.5 * I
    V1, V2, V3, V4, V5, V6 = [e[i] for i in range(2, 8)]
    return {
        "I": I, "II": II, "III": III,
        "aVR": aVR, "aVL": aVL, "aVF": aVF,
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6,
    }


def test_full_q8_identity():
    """Gate A test: Full Q8 = I_8 must return exact identity B = I_8, C = I_8."""
    I8 = np.eye(8, dtype=np.float64)
    T = 256
    rng = np.random.default_rng(42)
    X = rng.normal(size=(T, 8))
    Y = X @ I8.T  # Y = X

    B, Y_canon, C = canonicalize_measurements(I8, Y)
    
    assert B.shape == (8, 8)
    assert np.allclose(B, I8, atol=1e-12)
    assert np.allclose(C, I8, atol=1e-12)
    assert np.allclose(Y_canon, Y, atol=1e-12)


@pytest.mark.parametrize("subset_indices", [
    [0, 1, 2, 3, 4, 5],     # S6 precordial
    [0, 1, 2],              # S3 ICU
    [0, 1],                 # S2 Bipolar
    [0],                    # S1 Smartwatch
    [1, 5, 7],              # Non-contiguous
])
def test_canonical_subsets_identity(subset_indices):
    """Canonical coordinate subsets must be exactly preserved: B_Q == Q, Y_canon == Y."""
    I8 = np.eye(8, dtype=np.float64)
    Q_sub = I8[subset_indices]
    r = len(subset_indices)
    
    T = 256
    rng = np.random.default_rng(123)
    X = rng.normal(size=(T, 8))
    Y = X @ Q_sub.T

    B, Y_canon, C = canonicalize_measurements(Q_sub, Y)

    assert B.shape == (r, 8)
    assert np.allclose(B, Q_sub, atol=1e-12)
    assert np.allclose(C, np.eye(r), atol=1e-12)
    assert np.allclose(Y_canon, Y, atol=1e-12)


def test_six_limb_leads_rank2_reduction():
    """Six limb leads span rank 2; must canonicalize to exactly [e_I, e_II]."""
    ops = get_standard_operators()
    Q_limb = np.stack([ops["I"], ops["II"], ops["III"], ops["aVR"], ops["aVL"], ops["aVF"]], axis=0)
    
    T = 256
    rng = np.random.default_rng(42)
    X = rng.normal(size=(T, 8))
    Y = X @ Q_limb.T  # [T, 6]

    B, Y_canon, C = canonicalize_measurements(Q_limb, Y)

    # Rank must be exactly 2
    assert B.shape == (2, 8)
    e = np.eye(8, dtype=np.float64)
    assert np.allclose(B, e[:2], atol=1e-10)
    # The canonical waveforms must match real physical [I, II]
    expected_Y = X @ e[:2].T
    assert np.allclose(Y_canon, expected_Y, atol=1e-10)


def test_derived_limb_leads_rank2_recovery():
    """Inputting only derived limb leads {III, aVR, aVL, aVF} must recover [I, II]."""
    ops = get_standard_operators()
    Q_derived = np.stack([ops["III"], ops["aVR"], ops["aVL"], ops["aVF"]], axis=0)

    T = 256
    rng = np.random.default_rng(99)
    X = rng.normal(size=(T, 8))
    Y = X @ Q_derived.T  # [T, 4]

    B, Y_canon, C = canonicalize_measurements(Q_derived, Y)

    assert B.shape == (2, 8)
    e = np.eye(8, dtype=np.float64)
    assert np.allclose(B, e[:2], atol=1e-10)
    expected_Y = X @ e[:2].T
    assert np.allclose(Y_canon, expected_Y, atol=1e-10)


def test_all16_leads_redundancy_immunity():
    """Gate B: All 16 leads (8 independent + 8 derived) must canonicalize to exact Q8."""
    e = np.eye(8, dtype=np.float64)
    ops = get_standard_operators()
    q_16 = np.stack([
        e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7],
        ops["III"], ops["aVR"], ops["aVL"], ops["aVF"],
        (e[4] - e[3]) / np.sqrt(2),
        (e[2] - e[7]) / np.sqrt(2),
        (e[0] + e[5]) / np.sqrt(2),
        (e[1] - e[6]) / np.sqrt(2),
    ], axis=0)

    T = 256
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(T, 8))
    Y_16 = X @ q_16.T
    Y_8 = X @ e.T

    B, Y_canon, C = canonicalize_measurements(q_16, Y_16)

    assert B.shape == (8, 8)
    assert np.allclose(B, e, atol=1e-10)
    assert np.allclose(Y_canon, Y_8, atol=1e-10)


def test_basis_change_invariance():
    """Invertible linear transformation Q' = R Q, Y' = Y R^T must produce identical (B, Y_canon)."""
    rng = np.random.default_rng(777)
    # 5-lead subset
    e = np.eye(8, dtype=np.float64)
    Q = e[[0, 1, 3, 5, 7]]  # [5, 8]
    m = len(Q)

    # Random well-conditioned invertible R (5x5)
    A = rng.normal(size=(m, m))
    U, _, Vt = np.linalg.svd(A)
    R = U @ np.diag([1.2, 0.9, 1.5, 0.8, 1.1]) @ Vt

    Q_prime = R @ Q

    T = 256
    X = rng.normal(size=(T, 8))
    Y = X @ Q.T
    Y_prime = X @ Q_prime.T  # equals Y @ R.T

    B, Y_canon, _ = canonicalize_measurements(Q, Y)
    B_prime, Y_canon_prime, _ = canonicalize_measurements(Q_prime, Y_prime)

    assert np.allclose(B, B_prime, atol=1e-10)
    assert np.allclose(Y_canon, Y_canon_prime, atol=1e-10)


def test_permutation_and_sign_invariance():
    """Permuting rows and negating channels consistently must yield identical canonical outputs."""
    e = np.eye(8, dtype=np.float64)
    Q = e[[0, 1, 4]]  # Leads I, II, V3
    
    # Permute and negate
    P_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    Q_trans = P_mat @ Q

    T = 256
    rng = np.random.default_rng(888)
    X = rng.normal(size=(T, 8))
    Y = X @ Q.T
    Y_trans = X @ Q_trans.T

    B, Y_canon, _ = canonicalize_measurements(Q, Y)
    B_trans, Y_canon_trans, _ = canonicalize_measurements(Q_trans, Y_trans)

    assert np.allclose(B, B_trans, atol=1e-12)
    assert np.allclose(Y_canon, Y_canon_trans, atol=1e-12)


def test_subspace_continuity():
    """Check continuity of B along a smooth trajectory between coordinate axes."""
    # Rotate between coordinate 0 (Lead I) and coordinate 1 (Lead II)
    # q(alpha) = cos(alpha) e_0 + sin(alpha) e_1
    e = np.eye(8, dtype=np.float64)
    
    alphas = np.linspace(0.01, np.pi / 2 - 0.01, 50)
    diffs = []
    
    for i in range(len(alphas) - 1):
        a1 = alphas[i]
        a2 = alphas[i + 1]
        
        q1 = (np.cos(a1) * e[0] + np.sin(a1) * e[1]).reshape(1, 8)
        q2 = (np.cos(a2) * e[0] + np.sin(a2) * e[1]).reshape(1, 8)
        
        P1, _ = observable_projector(q1)
        P2, _ = observable_projector(q2)
        
        B1 = canonical_frame(P1)
        B2 = canonical_frame(P2)
        
        diff = np.linalg.norm(B2 - B1)
        diffs.append(diff)
        
    max_step_diff = max(diffs)
    # Ensure smooth transition along the interior of the arc
    assert max_step_diff < 0.1, f"Step diff too large: {max_step_diff}"
