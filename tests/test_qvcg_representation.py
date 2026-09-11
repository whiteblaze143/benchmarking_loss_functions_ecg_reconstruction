"""Unit Tests for Q-VCG Representation Extraction Contracts (PRD Section 68)."""
import numpy as np
import pytest

from rep_stat_ecg.src.motifs.representation import (
    extract_r0_vcg_summary,
    extract_r1_spatial_domain_features,
    extract_r2_quotient_motif_features,
    extract_r3_quotient_plus_where,
    compute_motif_occupancy_and_dwell,
    compute_transition_matrix,
)
from rep_stat_ecg.src.motifs.spatial_domains import CoordinateStandardizer, VoxelGrid3D
from rep_stat_ecg.src.motifs.quotient import QuotientClosure
from rep_stat_ecg.src.motifs.tokenizer import QVCGTokenizer


def test_occupancy_sums_to_one():
    """Verify occupancy vector sums strictly to 1.0 (including OOD slot)."""
    num_classes = 5
    tokens = np.array([0, 1, 1, 2, 4, 5, 5])  # 5 is OOD
    duration = 10.0

    occ, dwell = compute_motif_occupancy_and_dwell(tokens, num_classes, duration)
    assert len(occ) == num_classes + 1
    assert np.isclose(np.sum(occ), 1.0, atol=1e-6), f"Occupancy sum != 1.0: {np.sum(occ)}"
    # Verify OOD occupancy
    assert np.isclose(occ[5], 2.0 / 7.0, atol=1e-5)


def test_dwell_uses_physical_time():
    """Verify dwell is measured in physical seconds and sums to record duration."""
    num_classes = 4
    tokens = np.array([0, 0, 1, 2, 3])  # 5 samples
    duration_s = 2.5

    occ, dwell = compute_motif_occupancy_and_dwell(tokens, num_classes, duration_s)
    # Total dwell should match total duration
    assert np.isclose(np.sum(dwell), duration_s, atol=1e-6)
    # Token 0 appears 2 times -> 2/5 * 2.5 = 1.0 second
    assert np.isclose(dwell[0], 1.0, atol=1e-6)


def test_transition_counts_known_sequence():
    """Verify transition matrix correctly counts state transitions."""
    num_classes = 3
    # Sequence: 0 -> 1 -> 2 -> 0 -> 1
    tokens = np.array([0, 1, 2, 0, 1])

    T = compute_transition_matrix(tokens, num_classes, run_compress=False)
    # From state 0, moves to state 1 twice (100% of transitions from 0 go to 1)
    assert np.isclose(T[0, 1], 1.0)
    # From state 1, moves to state 2 once (100% of transitions from 1 go to 2)
    assert np.isclose(T[1, 2], 1.0)
    # From state 2, moves to state 0 once (100% of transitions from 2 go to 0)
    assert np.isclose(T[2, 0], 1.0)


def test_run_compression():
    """Verify run-compression collapses consecutive identical tokens."""
    num_classes = 3
    # Uncompressed: 0, 0, 0, 1, 1, 2
    # Compressed: 0, 1, 2
    tokens = np.array([0, 0, 0, 1, 1, 2])

    T_std = compute_transition_matrix(tokens, num_classes, run_compress=False)
    # In uncompressed, 0 -> 0 occurs twice
    assert T_std[0, 0] > 0.0

    T_run = compute_transition_matrix(tokens, num_classes, run_compress=True)
    # In run-compressed, self transitions are eliminated
    assert np.isclose(T_run[0, 0], 0.0)
    assert np.isclose(T_run[0, 1], 1.0)


def test_motif_and_where_separate_features():
    """Verify R3 cleanly factorizes Z_motif and Z_where."""
    num_motifs = 4
    T = 100
    rng = np.random.RandomState(42)
    v_physical = rng.normal(size=(3, T)).astype(np.float32)
    tokens = rng.randint(0, num_motifs, size=T)

    z_r3, z_motif, z_where = extract_r3_quotient_plus_where(
        v_physical, tokens, num_motifs, total_duration_s=10.0
    )

    # Dimensionality checks
    assert len(z_r3) == len(z_motif) + len(z_where)
    assert len(z_where) == num_motifs * 3
    assert np.allclose(z_r3[:len(z_motif)], z_motif)
    assert np.allclose(z_r3[len(z_motif):], z_where)


def test_ood_not_silently_remapped():
    """Verify out-of-support coordinates are flagged strictly as OOD."""
    std = CoordinateStandardizer()
    train_v = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1],
    ])
    std.fit(train_v)

    grid = VoxelGrid3D(grid_dim=10, coord_min=-2.0, coord_max=2.0)
    # Mark only center voxel as eligible
    grid.eligible_voxels = {grid.grid_dim ** 2 * 5 + grid.grid_dim * 5 + 5}
    grid.voxel_to_domain = {list(grid.eligible_voxels)[0]: 0}

    qc = QuotientClosure(n_domains=1, epsilon_mmd=0.01)
    qc.domain_to_motif = {0: 0}
    qc.motif_to_domains = {0: [0]}

    tok = QVCGTokenizer(standardizer=std, grid=grid, quotient=qc)

    # Point far outside grid: (100, 100, 100) with shape [3, 1]
    far_point = np.array([[100.0], [100.0], [100.0]], dtype=np.float32)  # [3, 1]
    tokens, ood_rate = tok.tokenize(far_point)

    assert tokens[0] == tok.ood_token, f"Far point was silently remapped to {tokens[0]}!"
    assert np.isclose(ood_rate, 1.0)
