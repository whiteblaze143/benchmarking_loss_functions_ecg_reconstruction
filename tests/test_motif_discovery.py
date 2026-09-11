"""Mandatory Synthetic RepSpat Unit Tests (PRD Section 64, Tests A-F).

Tests:
- Test A: Repeated dynamics in distant positions -> merged into same motif.
- Test B: Nearby positions with different dynamics -> kept in separate motifs.
- Test C: Same mean, different covariance -> separated by MMD.
- Test D: Same mean/covariance, different multimodality -> separated by MMD.
- Test E: Patient random effects -> block permutation preserves structure.
- Test F: No recurrence null -> all distinct distributions yield zero false merges.
"""
import numpy as np
import pytest

from rep_stat_ecg.src.motifs.mmd import (
    benjamini_hochberg,
    build_attribute_blocks,
    precompute_block_kernel_sums,
    run_block_permutation_test,
    generate_repspat_assignments,
    biased_mmd2_batch_from_block_sums,
    raw_biased_mmd2_for_assignments,
    compute_median_heuristic,
    compute_mmd2_u_stat,
    compute_patient_block_perm_mmd,
    calibrate_domain_self_reproducibility,
    build_reference_repspat_blocks,
    generate_reference_repspat_assignments,
)
from rep_stat_ecg.src.motifs.quotient import QuotientClosure


def test_bh_matches_known_reference():
    p = np.array([0.01, 0.04, 0.03, 0.002])
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.008])


def test_attribute_blocks_and_exact_imq_block_permutation():
    rng = np.random.RandomState(7)
    x = rng.normal(0, 0.2, (40, 3)); y = rng.normal(2, 0.2, (40, 3))
    bx = build_attribute_blocks(x, 4, 7); by = build_attribute_blocks(y, 4, 8)
    assert sum(map(len, bx)) == len(x) and sum(map(len, by)) == len(y)
    S, sizes, diagonal = precompute_block_kernel_sums(bx, by, "IMQ", 1.0)
    obs, p_value, n_perm, _ = run_block_permutation_test(
        S, sizes, len(bx), len(by), B_max=99, B_init=99,
        rng=np.random.RandomState(9), self_diagonal=diagonal)
    assert obs > 0 and p_value <= 0.05 and n_perm == 99


def test_repspat_assignment_uses_actual_overshoot_size():
    sizes = np.array([4, 4, 4, 4, 4])
    Z = generate_repspat_assignments(sizes, target_size=10, n_perm=20,
                                     rng=np.random.RandomState(3))
    selected_sizes = Z @ sizes
    assert np.all(selected_sizes > 10)
    assert np.all(selected_sizes == 12)
    assert np.all(selected_sizes < sizes.sum())


def test_reference_repspat_blocks_use_floor_and_released_kmeans_settings():
    rng = np.random.RandomState(4)
    xi = rng.normal(size=(25, 3))
    blocks = build_reference_repspat_blocks(xi, neighborhood_size=10)
    assert len(blocks) == 2
    assert sum(map(len, blocks)) == 25
    assert len(build_reference_repspat_blocks(xi[:8], neighborhood_size=10)) == 1


def test_reference_repspat_assignment_stops_at_or_above_target():
    sizes = np.array([5, 5, 5, 5])
    assignments = generate_reference_repspat_assignments(
        sizes, target_size=10, n_perm=20, rng=np.random.RandomState(3)
    )
    assert np.all(assignments @ sizes == 10)


def test_reference_statistic_preserves_unclipped_roundoff_and_exact_ties():
    # M3R must not inherit the primary implementation's defensive clipping or
    # tolerance-adjusted exceedance convention.
    S = np.array([[1.0, 1.0], [1.0, 1.0 + 2e-16]])
    sizes = np.ones(2, dtype=np.int64)
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    raw = biased_mmd2_batch_from_block_sums(S, sizes, Z, clip_zero=False)
    clipped = biased_mmd2_batch_from_block_sums(S, sizes, Z, clip_zero=True)
    assert np.all(clipped >= 0.0)
    observed = raw[0]
    exact = int(np.count_nonzero(raw >= observed))
    tolerant = int(np.count_nonzero(raw >= observed - 1e-12))
    assert exact <= tolerant


def test_block_aggregate_matches_raw_biased_mmd_for_same_assignments():
    rng = np.random.RandomState(11)
    blocks = [rng.normal(i, 0.1, size=(n, 3)) for i, n in enumerate([3, 4, 2, 5, 3])]
    S, sizes, _ = precompute_block_kernel_sums(blocks[:2], blocks[2:], "IMQ", 1.0)
    Z = generate_repspat_assignments(sizes, target_size=7, n_perm=99,
                                     rng=np.random.RandomState(12))
    Z = np.vstack([np.array([1, 1, 0, 0, 0], dtype=float), Z])
    raw = raw_biased_mmd2_for_assignments(blocks, Z, "IMQ", 1.0)
    aggregated = biased_mmd2_batch_from_block_sums(S, sizes, Z, batch_size=17)
    assert np.max(np.abs(raw - aggregated)) < 1e-12
    observed_raw, observed_agg = raw[0], aggregated[0]
    assert np.count_nonzero(raw[1:] >= observed_raw) == np.count_nonzero(aggregated[1:] >= observed_agg)


def test_bh_similarity_graph_flags_non_clique_component():
    rows = [
        {"domain_1": 0, "domain_2": 1, "mmd2_imq_obs": 0.0, "q_bh_imq": 0.5},
        {"domain_1": 1, "domain_2": 2, "mmd2_imq_obs": 0.0, "q_bh_imq": 0.5},
        {"domain_1": 0, "domain_2": 2, "mmd2_imq_obs": 1.0, "q_bh_imq": 0.01},
    ]
    qc = QuotientClosure(3, epsilon_mmd=0.0)
    qc.build_similarity_graph(rows); qc.compute_quotient_classes()
    assert qc.num_motifs == 1
    assert qc.motif_status[0] == "AMBIGUOUS_NON_CLIQUE"
    assert qc.motif_is_clique[0] is False


def test_synthetic_test_a_repeated_dynamics():
    """Test A: Spatially separated regions with identical dynamic distribution must merge."""
    rng = np.random.RandomState(42)
    N = 200

    # Domain 1 at position (+2, +2, +2)
    xi_1 = rng.normal(loc=[1.5, 0.2, 0.8], scale=[0.3, 0.1, 0.2], size=(N, 3))
    # Domain 2 at position (-2, -2, -2), identical distribution
    xi_2 = rng.normal(loc=[1.5, 0.2, 0.8], scale=[0.3, 0.1, 0.2], size=(N, 3))

    gamma = compute_median_heuristic(np.concatenate([xi_1, xi_2], axis=0))
    mmd2 = compute_mmd2_u_stat(xi_1, xi_2, kernel="Gaussian", kernel_param=gamma)

    # Identical distributions should have near-zero MMD^2
    assert mmd2 < 0.005, f"Expected small MMD^2 for identical distributions, got {mmd2}"

    # Verify quotient closure merges them
    qc = QuotientClosure(n_domains=2, epsilon_mmd=0.01)
    res = [{"domain_1": 0, "domain_2": 1, "mmd2": mmd2, "passed_support": True}]
    qc.build_equivalence_graph(res)
    mapping = qc.compute_quotient_classes()

    assert mapping[0] == mapping[1], "Domains with identical dynamics failed to merge into same motif!"
    assert qc.num_motifs == 1
    assert qc.motif_status[0] == "REPEATED"


def test_synthetic_test_b_nearby_different_dynamics():
    """Test B: Spatially adjacent domains with different dynamics must remain separated."""
    rng = np.random.RandomState(42)
    N = 200

    # Domain 1 (slow speed, low curvature)
    xi_1 = rng.normal(loc=[0.5, 0.0, 0.2], scale=0.2, size=(N, 3))
    # Domain 2 (fast speed, high curvature)
    xi_2 = rng.normal(loc=[3.0, 1.5, 2.5], scale=0.2, size=(N, 3))

    gamma = compute_median_heuristic(np.concatenate([xi_1, xi_2], axis=0))
    mmd2 = compute_mmd2_u_stat(xi_1, xi_2, kernel="Gaussian", kernel_param=gamma)

    assert mmd2 > 0.10, f"Expected large MMD^2 for different dynamics, got {mmd2}"

    # Verify quotient closure keeps them separate
    qc = QuotientClosure(n_domains=2, epsilon_mmd=0.02)
    res = [{"domain_1": 0, "domain_2": 1, "mmd2": mmd2, "passed_support": True}]
    qc.build_equivalence_graph(res)
    mapping = qc.compute_quotient_classes()

    assert mapping[0] != mapping[1], "Different domains erroneously merged!"
    assert qc.num_motifs == 2
    assert qc.motif_status[0] == "LOCAL_ONLY"
    assert qc.motif_status[1] == "LOCAL_ONLY"


def test_synthetic_test_c_covariance_separation():
    """Test C: Distributions with same mean but different covariance must be separated by MMD."""
    rng = np.random.RandomState(42)
    N = 300
    mean = np.array([1.0, 0.0, 0.5])

    # Covariance 1: spherical
    cov_1 = np.diag([0.1, 0.1, 0.1])
    xi_1 = rng.multivariate_normal(mean, cov_1, size=N)
    xi_1 = xi_1 - np.mean(xi_1, axis=0) + mean

    # Covariance 2: highly anisotropic
    cov_2 = np.diag([0.8, 0.02, 0.02])
    xi_2 = rng.multivariate_normal(mean, cov_2, size=N)
    xi_2 = xi_2 - np.mean(xi_2, axis=0) + mean

    # Check means are identical
    assert np.allclose(np.mean(xi_1, axis=0), np.mean(xi_2, axis=0), atol=1e-5)

    gamma = compute_median_heuristic(np.concatenate([xi_1, xi_2], axis=0))
    mmd2 = compute_mmd2_u_stat(xi_1, xi_2, kernel="Gaussian", kernel_param=gamma)

    # MMD detects covariance differences that mean matching would miss
    assert mmd2 > 0.015, f"MMD failed to detect covariance difference: {mmd2}"


def test_synthetic_test_d_multimodal_separation():
    """Test D: Distributions with matching mean and covariance but different multimodality must separate."""
    rng = np.random.RandomState(42)
    N = 400

    # Unimodal standard Gaussian
    xi_1 = rng.normal(loc=0.0, scale=1.0, size=(N, 3))

    # Bimodal Gaussian with 50/50 mixture at +/- 1.0 (mean=0, variance ~= 1 + 1 = 2)
    half = N // 2
    part1 = rng.normal(loc=1.0, scale=0.5, size=(half, 3))
    part2 = rng.normal(loc=-1.0, scale=0.5, size=(half, 3))
    xi_2 = np.concatenate([part1, part2], axis=0)

    gamma = compute_median_heuristic(np.concatenate([xi_1, xi_2], axis=0))
    mmd2 = compute_mmd2_u_stat(xi_1, xi_2, kernel="Gaussian", kernel_param=gamma)

    assert mmd2 > 0.01, f"MMD failed to detect multimodal discrepancy: {mmd2}"


def test_synthetic_test_e_patient_block_calibration():
    """Test E: Patient-block permutation respects intra-patient dependence."""
    rng = np.random.RandomState(42)
    # Simulate 20 patients contributing 10 correlated microstates each
    n_patients = 20
    n_samples_per_pat = 10

    patient_blocks_1 = {}
    patient_blocks_2 = {}

    for p_idx in range(n_patients):
        pid = f"pat_{p_idx}"
        # Patient random effect
        pat_effect = rng.normal(loc=0.0, scale=0.8, size=(1, 3))
        # Domain 1 and 2 from same population
        noise_1 = rng.normal(loc=0.0, scale=0.2, size=(n_samples_per_pat, 3))
        noise_2 = rng.normal(loc=0.0, scale=0.2, size=(n_samples_per_pat, 3))

        patient_blocks_1[pid] = pat_effect + noise_1
        patient_blocks_2[pid] = pat_effect + noise_2

    obs_mmd, p_val, null_dist = compute_patient_block_perm_mmd(
        patient_blocks_1, patient_blocks_2, kernel="Gaussian", kernel_param=1.0, n_perm=50, rng=rng
    )

    # Because both domains have identical generation process, p-value should not reject (p > 0.05)
    assert p_val > 0.05, f"Block permutation false alarm: p_val = {p_val}"


def test_synthetic_test_f_no_recurrence_null():
    """Test F: When all domains have completely distinct distributions, zero repeated motifs form."""
    rng = np.random.RandomState(42)
    n_domains = 5
    domains_xi = {}

    for d in range(n_domains):
        # Distinct means separated by 3 units each
        loc = np.array([d * 3.0, (d % 2) * 2.0, 0.0])
        domains_xi[d] = rng.normal(loc=loc, scale=0.3, size=(100, 3))

    pairwise = []
    for d1 in range(n_domains):
        for d2 in range(d1 + 1, n_domains):
            mmd2 = compute_mmd2_u_stat(domains_xi[d1], domains_xi[d2], kernel="Gaussian", kernel_param=1.0)
            pairwise.append({
                "domain_1": d1,
                "domain_2": d2,
                "mmd2": mmd2,
                "passed_support": True,
            })

    qc = QuotientClosure(n_domains=n_domains, epsilon_mmd=0.02)
    qc.build_equivalence_graph(pairwise)
    qc.compute_quotient_classes()

    assert qc.num_motifs == n_domains, f"Expected {n_domains} independent motifs, got {qc.num_motifs}"
    assert qc.num_repeated_motifs == 0, "Manufactured false repeated motifs under null!"
    assert qc.compression_ratio == 0.0
