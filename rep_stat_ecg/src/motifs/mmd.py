"""MMD with dependence-preserving block permutation and BH FDR correction.

Faithful adaptation of repSpat (Senanayake & Jeganathan) to VCG phase space.

Key design:
- Primary kernel: IMQ with c²=1 (per repSpat Monte Carlo analysis).
- Blocks: K-means partition of the attribute space (ξ = (s, ρ, κ)) within each domain,
  following repSpat Algorithm 1 / Appendix.
- Permutation: randomly reassign whole blocks between the two domains (block sizes preserved).
- Vectorized: pairwise block kernel sums precomputed once; permutations are index arithmetic.
- BH FDR at α=0.05 over all pairwise tests (exact repSpat procedure).
- Edge semantics: insufficient evidence to conclude distributions differ (NOT statistical equivalence).

Additional kernels (Gaussian, Matérn52, Laplacian, RQ, Polynomial) are available
for later sensitivity analyses but do NOT participate in the primary quotient.
"""
from __future__ import annotations

import numpy as np
import scipy.spatial.distance as sp_dist


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

#: Primary inference kernel (IMQ, c²=1).
PRIMARY_KERNEL: str = "IMQ"
PRIMARY_KERNEL_PARAM: float = 1.0

#: All available kernels (primary first, then sensitivity, then exploratory).
KERNEL_REGISTRY: list[str] = [
    "IMQ",          # primary — repSpat default
    "Gaussian",     # sensitivity A
    "Matern52",     # sensitivity B
    "Laplacian",    # exploratory
    "RQ",           # exploratory
    "Polynomial",   # exploratory (NOT characteristic — diagnostic only)
]


# ---------------------------------------------------------------------------
# Kernel matrix computation
# ---------------------------------------------------------------------------

def compute_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
) -> np.ndarray:
    """Kernel Gram matrix K[i,j] = k(X[i], Y[j]).

    All kernels parameterized by kernel_param (bandwidth / c²).
    """
    if Y is None:
        Y = X

    # Compute squared distances once (shared by most kernels)
    if kernel in ("IMQ", "Gaussian", "Matern52", "RQ"):
        dists_sq = sp_dist.cdist(X, Y, metric="sqeuclidean")

    if kernel == "IMQ":
        # k(x,y) = (c² + ||x-y||²)^{-1/2};  repSpat default: c²=1
        return 1.0 / np.sqrt(max(kernel_param, 1e-9) + dists_sq)

    elif kernel == "Gaussian":
        return np.exp(-dists_sq / max(kernel_param, 1e-9))

    elif kernel == "Matern52":
        dists = np.sqrt(np.clip(dists_sq, 0.0, None))
        r = dists / max(float(np.sqrt(kernel_param)), 1e-9)
        sqrt5_r = np.sqrt(5.0) * r
        return (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * np.exp(-sqrt5_r)

    elif kernel == "Laplacian":
        dists_l1 = sp_dist.cdist(X, Y, metric="cityblock")
        return np.exp(-dists_l1 / max(kernel_param, 1e-9))

    elif kernel == "RQ":
        alpha = 2.0
        return (1.0 + dists_sq / (2.0 * alpha * max(kernel_param, 1e-9))) ** (-alpha)

    elif kernel == "Polynomial":
        dot = X @ Y.T
        return (1.0 + dot / max(kernel_param, 1e-9)) ** 3

    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Options: {KERNEL_REGISTRY}")


def compute_median_heuristic(
    X: np.ndarray,
    sample_size: int = 1000,
    random_state: int = 42,
) -> float:
    """Median pairwise squared-Euclidean distance (for bandwidth selection)."""
    if len(X) <= 1:
        return 1.0
    rng = np.random.RandomState(random_state)
    if len(X) > sample_size:
        X = X[rng.choice(len(X), size=sample_size, replace=False)]
    dists_sq = sp_dist.pdist(X, metric="sqeuclidean")
    med = float(np.median(dists_sq))
    return med if med > 1e-9 else 1.0


# ---------------------------------------------------------------------------
# Attribute-based block construction (per repSpat appendix)
# ---------------------------------------------------------------------------

def build_attribute_blocks(
    xi: np.ndarray,
    n_blocks: int,
    random_state: int = 42,
) -> list[np.ndarray]:
    """Partition domain microstates into attribute-space blocks via K-means.

    Follows repSpat Algorithm 1 Appendix: partition each cluster into B_c blocks
    using K-means on the attribute (feature) space. Blocks are then the unit
    of permutation — whole blocks are swapped between domains.

    Args:
        xi: Array of shape (N, d) — the d-dimensional microstate descriptors
            within one domain (for Q-VCG: d=3, columns = s, ρ, κ).
        n_blocks: Number of K-means blocks. repSpat uses ceil(|X_c| / L)
            where L is the CAHC neighborhood parameter.
        random_state: For K-means reproducibility.

    Returns:
        List of arrays, each of shape (n_k, d), one per non-empty block.
    """
    from sklearn.cluster import MiniBatchKMeans

    N = len(xi)
    if N == 0:
        return []

    n_blocks = min(n_blocks, N)  # can't have more blocks than points
    if n_blocks <= 1:
        return [xi]

    km = MiniBatchKMeans(
        n_clusters=n_blocks,
        random_state=random_state,
        n_init=3,
        max_iter=100,
    )
    labels = km.fit_predict(xi)

    blocks = []
    for b in range(n_blocks):
        mask = labels == b
        if mask.sum() > 0:
            blocks.append(xi[mask])
    return blocks


def compute_n_blocks(n_microstates: int, cahc_neighborhood_size: int = 10) -> int:
    """Compute block count following repSpat: B_c = ceil(|X_c| / L).

    Args:
        n_microstates: Number of microstates in domain.
        cahc_neighborhood_size: CAHC local neighborhood parameter L.

    Returns:
        Number of blocks (minimum 2).
    """
    import math
    return max(2, math.ceil(n_microstates / max(cahc_neighborhood_size, 1)))


def build_reference_repspat_blocks(
    xi: np.ndarray,
    neighborhood_size: int = 10,
) -> list[np.ndarray]:
    """Released-repSpat block construction for the M3R sensitivity run.

    This deliberately mirrors ``repspat.clustering.create_blocks``: use
    ``floor(n / m)``, collapse to one block when that value is zero or is not
    smaller than the number of unique attribute rows, and otherwise run
    standard KMeans with ``n_init=10`` and ``random_state=0``.
    """
    from sklearn.cluster import KMeans

    xi = np.asarray(xi, dtype=np.float64)
    if len(xi) == 0:
        return []
    n_blocks = len(xi) // max(int(neighborhood_size), 1)
    n_unique = len(np.unique(xi, axis=0))
    if n_blocks == 0 or n_blocks >= n_unique:
        return [xi]
    labels = KMeans(n_clusters=n_blocks, n_init=10, random_state=0).fit_predict(xi)
    return [xi[labels == block_id] for block_id in range(n_blocks)]


# ---------------------------------------------------------------------------
# Block kernel sum cache — the key vectorization
# ---------------------------------------------------------------------------

def precompute_block_kernel_sums(
    blocks_i: list[np.ndarray],
    blocks_j: list[np.ndarray],
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes the block×block kernel sum matrix for a domain pair.

    All microstates from both domains are concatenated; pairwise distances
    are computed ONCE; kernel is applied once; sums are aggregated into a
    (Q_i + Q_j) × (Q_i + Q_j) block sum matrix S.

    S[a, b] = Σ_{x ∈ block_a} Σ_{y ∈ block_b} k(x, y)

    With S cached, any block reassignment can compute its exact MMD² via
    index arithmetic over S — no raw microstate access needed in permutations.

    Args:
        blocks_i: List of microstate arrays for domain G_i.
        blocks_j: List of microstate arrays for domain G_j.
        kernel: Kernel name.
        kernel_param: Bandwidth / c².

    Returns:
        S: float64 array of shape (Q, Q), Q = Q_i + Q_j.
        block_sizes: int64 array of length Q (number of microstates per block).
    """
    Q_i = len(blocks_i)
    Q_j = len(blocks_j)
    Q = Q_i + Q_j

    all_blocks = blocks_i + blocks_j
    block_sizes = np.array([len(b) for b in all_blocks], dtype=np.int64)

    # Concatenate all microstates. Compute in row chunks so exact point-level
    # kernel sums do not require materializing an N x N Gram matrix.
    all_xi = np.concatenate(all_blocks, axis=0)

    # Build cumulative index boundaries
    ends = np.cumsum(block_sizes)
    starts = np.concatenate([[0], ends[:-1]])

    S = np.zeros((Q, Q), dtype=np.float64)
    row_chunk = 512
    for a, (start, end) in enumerate(zip(starts, ends)):
        for lo in range(start, end, row_chunk):
            K_chunk = compute_kernel_matrix(all_xi[lo:min(lo + row_chunk, end)], all_xi,
                                            kernel=kernel, kernel_param=kernel_param)
            S[a] += np.add.reduceat(K_chunk, starts, axis=1).sum(axis=0)

    # All registered kernels have k(x,x)=1, including IMQ with c²=1.
    self_diagonal = block_sizes.astype(np.float64)
    return S, block_sizes, self_diagonal


# ---------------------------------------------------------------------------
# MMD² from block kernel sums (biased Eq. 6 statistic used by repSpat)
# ---------------------------------------------------------------------------

def mmd2_from_block_sums(
    S: np.ndarray,
    block_sizes: np.ndarray,
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    self_diagonal: np.ndarray | None = None,
) -> float:
    """Exact biased empirical MMD² from precomputed block kernel sums.

    Computes:
        MMD²(G_i, G_j) =
            Σ_{a≠b ∈ G_i} S[a,b] / (n_i(n_i-1))
          + Σ_{a≠b ∈ G_j} S[a,b] / (n_j(n_j-1))
          - 2 Σ_{a ∈ G_i, b ∈ G_j} S[a,b] / (n_i · n_j)

    where n_i, n_j are total microstate counts (sum of block_sizes in each group).

    This is EXACTLY equivalent to computing the U-statistic MMD² directly on
    all raw microstates; the block-sum decomposition is a computational identity.

    Args:
        S: (Q, Q) block kernel sum matrix.
        block_sizes: (Q,) number of microstates per block.
        mask_i: Boolean mask, blocks in group G_i.
        mask_j: Boolean mask, blocks in group G_j (disjoint from mask_i).

    Returns:
        Unbiased MMD² ≥ 0.
    """
    n_i = int(block_sizes[mask_i].sum())
    n_j = int(block_sizes[mask_j].sum())
    if n_i < 1 or n_j < 1:
        return 0.0

    idx_i = np.where(mask_i)[0]
    idx_j = np.where(mask_j)[0]

    S_ii = S[np.ix_(idx_i, idx_i)]
    S_jj = S[np.ix_(idx_j, idx_j)]
    S_ij = S[np.ix_(idx_i, idx_j)]

    term_ii = S_ii.sum() / (n_i * n_i)
    term_jj = S_jj.sum() / (n_j * n_j)
    term_ij = S_ij.sum() / (n_i * n_j)

    mmd2 = term_ii + term_jj - 2.0 * term_ij
    return float(max(0.0, mmd2))


# ---------------------------------------------------------------------------
# Fixed block permutations using the repSpat cumulative-overshoot rule
# ---------------------------------------------------------------------------

def generate_repspat_assignments(block_sizes: np.ndarray, target_size: int,
                                 n_perm: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate whole-block assignments until selected size strictly exceeds target."""
    q = len(block_sizes)
    Z = np.zeros((n_perm, q), dtype=np.float64)
    for r in range(n_perm):
        order = rng.permutation(q)
        cumulative = np.cumsum(block_sizes[order])
        count = min(int(np.searchsorted(cumulative, target_size, side="right")) + 1, q - 1)
        Z[r, order[:count]] = 1.0
    return Z


def generate_reference_repspat_assignments(
    block_sizes: np.ndarray,
    target_size: int,
    n_perm: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Mirror the released package's block draw loop (stop at ``>=`` target).

    Sampling a random remaining block repeatedly is distributionally identical
    to consuming a random permutation.  ``side='left'`` reproduces the package
    loop's ``while n_permuted < n_size`` stopping rule.
    """
    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    q = len(block_sizes)
    Z = np.zeros((n_perm, q), dtype=np.float64)
    for r in range(n_perm):
        order = rng.permutation(q)
        cumulative = np.cumsum(block_sizes[order])
        count = min(int(np.searchsorted(cumulative, target_size, side="left")) + 1, q - 1)
        Z[r, order[:count]] = 1.0
    return Z


def biased_mmd2_batch_from_block_sums(S: np.ndarray, block_sizes: np.ndarray,
                                      Z: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Exact Eq. 6 statistics for block assignments, batched through BLAS."""
    Z = np.asarray(Z, dtype=np.float64)
    result = np.empty(len(Z), dtype=np.float64)
    total_kernel_sum = float(S.sum())
    row_sums = S.sum(axis=1)
    total_n = float(block_sizes.sum())
    for start in range(0, len(Z), batch_size):
        z = Z[start:start + batch_size]
        n_x = z @ block_sizes
        n_y = total_n - n_x
        zk = z @ S
        s_xx = np.einsum("ij,ij->i", zk, z)
        selected_row_sum = z @ row_sums
        s_xy = selected_row_sum - s_xx
        s_yy = total_kernel_sum - s_xx - 2.0 * s_xy
        result[start:start + len(z)] = s_xx / n_x**2 + s_yy / n_y**2 - 2.0 * s_xy / (n_x * n_y)
    return np.maximum(result, 0.0)


def raw_biased_mmd2_for_assignments(blocks: list[np.ndarray], Z: np.ndarray,
                                    kernel: str = "IMQ", kernel_param: float = 1.0) -> np.ndarray:
    """Small-data reference implementation used only by the regression gate."""
    sizes = np.asarray([len(block) for block in blocks], dtype=np.int64)
    labels = np.repeat(np.arange(len(blocks)), sizes)
    X = np.concatenate(blocks)
    K = compute_kernel_matrix(X, X, kernel, kernel_param)
    out = []
    for z in np.asarray(Z, dtype=bool):
        in_x = z[labels]
        in_y = ~in_x
        n_x, n_y = int(in_x.sum()), int(in_y.sum())
        s_xx = K[np.ix_(in_x, in_x)].sum()
        s_yy = K[np.ix_(in_y, in_y)].sum()
        s_xy = K[np.ix_(in_x, in_y)].sum()
        out.append(s_xx / n_x**2 + s_yy / n_y**2 - 2.0 * s_xy / (n_x * n_y))
    return np.maximum(np.asarray(out), 0.0)


def count_permutation_exceedances(null: np.ndarray, observed: float,
                                  numerical_atol: float = 1e-12) -> int:
    """Count ties consistently despite harmless floating-point summation order."""
    return int(np.count_nonzero(np.asarray(null) >= float(observed) - numerical_atol))

def run_block_permutation_test(
    S: np.ndarray,
    block_sizes: np.ndarray,
    Q_i: int,
    Q_j: int,
    B_max: int = 999,
    B_init: int = 99,
    rng: np.random.RandomState | None = None,
    self_diagonal: np.ndarray | None = None,
) -> tuple[float, float, int, str]:
    """Run exactly B_max repSpat whole-block permutations (no adaptive stopping).

    Permutation scheme (faithful to repSpat):
        Randomly assign Q_i blocks to "permuted G_i" and Q_j blocks to
        "permuted G_j" from the pool of Q_i + Q_j blocks total.
        This preserves group sizes while randomizing which domain each block
        is associated with.

    Sequential stopping:
        After B_init permutations, assess whether additional permutations
        can change the outcome. Stop early if the p-value range is decisive.

    Args:
        S: (Q, Q) block kernel sum matrix, Q = Q_i + Q_j.
        block_sizes: (Q,) microstate counts per block.
        Q_i: Number of blocks in domain G_i.
        Q_j: Number of blocks in domain G_j.
        B_max: Maximum permutations (default 999 → p_min = 0.001).
        B_init: Initial permutations before adaptive check.
        rng: RandomState.

    Returns:
        obs_mmd2: Observed MMD² under original block assignment.
        p_value: Permutation p-value = (1 + #{T^b ≥ T^obs}) / (B + 1).
        n_perms: Actual permutations performed.
        stop_reason: "converged_different" | "converged_similar" | "max_reached".
    """
    if rng is None:
        rng = np.random.RandomState(42)

    Q = Q_i + Q_j
    mask_i_obs = np.zeros(Q, dtype=bool)
    mask_i_obs[:Q_i] = True
    mask_j_obs = ~mask_i_obs

    obs_mmd2 = mmd2_from_block_sums(S, block_sizes, mask_i_obs, mask_j_obs, self_diagonal)

    target_size = int(min(block_sizes[:Q_i].sum(), block_sizes[Q_i:].sum()))
    assignments = generate_repspat_assignments(block_sizes, target_size, B_max, rng)
    null = biased_mmd2_batch_from_block_sums(S, block_sizes, assignments)
    exceedances = count_permutation_exceedances(null, obs_mmd2)
    return obs_mmd2, (1 + exceedances) / (B_max + 1), B_max, "fixed_repspat"


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR correction (exact repSpat procedure)
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Benjamini–Hochberg FDR correction over all pairwise tests.

    Returns BH-adjusted q-values. A pair has a similarity edge when q ≥ alpha
    (null of equal distributions is NOT rejected).

    This is the exact multiple-comparison procedure used in repSpat.

    Args:
        p_values: Raw permutation p-values, shape (n_pairs,).
        alpha: FDR level (repSpat uses 0.05).

    Returns:
        q_values: BH-adjusted p-values, same shape.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1, dtype=float)

    # BH: q_k = min_{j≥k}(n * p_j / j), clipped at 1
    adjusted = np.minimum(1.0, sorted_p * n / ranks)
    # Enforce monotonicity (take cumulative min from right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    q_values = np.empty(n, dtype=float)
    q_values[order] = adjusted
    return q_values


# ---------------------------------------------------------------------------
# Simple point-estimate MMD (for diagnostics and domain statistics)
# ---------------------------------------------------------------------------

def compute_mmd2_u_stat(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
    max_samples: int = 2000,
    random_state: int = 42,
) -> float:
    """Unbiased U-statistic MMD² with deterministic subsampling cap.

    Used for diagnostics and domain statistics only.
    The block-permutation procedure is the primary inferential statistic.
    """
    rng = np.random.RandomState(random_state)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0

    K_xx = compute_kernel_matrix(X, X, kernel=kernel, kernel_param=kernel_param)
    K_yy = compute_kernel_matrix(Y, Y, kernel=kernel, kernel_param=kernel_param)
    K_xy = compute_kernel_matrix(X, Y, kernel=kernel, kernel_param=kernel_param)
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    mmd2 = (
        K_xx.sum() / (n * (n - 1))
        + K_yy.sum() / (m * (m - 1))
        - 2.0 * K_xy.sum() / (n * m)
    )
    return float(max(0.0, mmd2))


# Legacy helpers retained for callers outside the repSpat-primary M3 path.
def compute_patient_block_perm_mmd(patient_blocks_1, patient_blocks_2, kernel="IMQ",
                                   kernel_param=1.0, n_perm=99, rng=None, max_samples=2000):
    rng = np.random.RandomState(42) if rng is None else rng
    keys = sorted(set(patient_blocks_1) & set(patient_blocks_2))
    X = np.concatenate([patient_blocks_1[k] for k in keys])
    Y = np.concatenate([patient_blocks_2[k] for k in keys])
    observed = compute_mmd2_u_stat(X, Y, kernel, kernel_param, max_samples)
    null = []
    for _ in range(n_perm):
        left, right = [], []
        for key in keys:
            a, b = patient_blocks_1[key], patient_blocks_2[key]
            if rng.rand() < 0.5: a, b = b, a
            left.append(a); right.append(b)
        null.append(compute_mmd2_u_stat(np.concatenate(left), np.concatenate(right), kernel, kernel_param, max_samples))
    null = np.asarray(null)
    return observed, float((1 + np.count_nonzero(null >= observed)) / (n_perm + 1)), null


def calibrate_domain_self_reproducibility(domain_patient_blocks, kernel="IMQ", kernel_param=1.0,
                                           min_patients_for_split=8, quantile=0.95):
    values = []
    for blocks in domain_patient_blocks.values():
        keys = sorted(blocks)
        if len(keys) < min_patients_for_split:
            continue
        midpoint = len(keys) // 2
        X = np.concatenate([blocks[k] for k in keys[:midpoint]])
        Y = np.concatenate([blocks[k] for k in keys[midpoint:]])
        values.append(compute_mmd2_u_stat(X, Y, kernel, kernel_param))
    return (float(np.quantile(values, quantile)) if values else 0.0), values
