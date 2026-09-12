"""3D Vectorcardiographic (VCG) spatial adjacency matrix construction.

Implements Step 1.2 of the repSpat analogue in VCG space R^3:
- Spatial Adjacency Matrix L [n, n] constructed via m-nearest neighbors in R^3.
- Points from different cardiac cycles passing through similar regions of VCG space
  (e.g., recurrent QRS loops across beats) become spatial neighbors (l_ij = 1).
- Optional hybrid adjacency combining VCG-spatial and temporal contiguity.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def construct_vcg_adjacency(
    vcg: np.ndarray,
    m_neighbors: int = 4,
    symmetric: bool = True,
) -> sp.csr_matrix:
    """Constructs 3D VCG-space spatial adjacency matrix L of shape [n, n].

    Parameters
    ----------
    vcg : np.ndarray
        Array of shape [n, 3] containing 3D VCG dipole coordinates (Vx, Vy, Vz).
    m_neighbors : int
        Number of spatial nearest neighbors in R^3 (m >= 1).
    symmetric : bool
        Whether to enforce undirected/symmetric links: L = max(L, L^T).

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Binary sparse adjacency matrix of shape [n, n] with zero diagonal.
    """
    vcg_arr = np.asarray(vcg, dtype=np.float64)
    if vcg_arr.ndim != 2 or vcg_arr.shape[1] != 3:
        raise ValueError(f"Expected VCG coordinates of shape [n, 3], got {vcg_arr.shape}")

    n = len(vcg_arr)
    if n <= 1:
        return sp.csr_matrix((n, n), dtype=int)

    k = min(max(1, m_neighbors), n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(vcg_arr)
    # Exclude self (column 0)
    indices = nn.kneighbors(vcg_arr, return_distance=False)[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = np.ones(len(rows), dtype=int)

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if symmetric:
        L = L.maximum(L.T)

    L.setdiag(0)
    L.eliminate_zeros()
    return L


def construct_hybrid_adjacency(
    vcg: np.ndarray,
    m_neighbors: int = 4,
    temporal_window: int = 1,
    symmetric: bool = True,
) -> sp.csr_matrix:
    """Constructs hybrid adjacency combining 3D VCG proximity with temporal contiguity.

    Two time samples i and j are connected if s(j) is among the m-nearest VCG neighbors
    of s(i) OR if |i - j| <= temporal_window.

    Parameters
    ----------
    vcg : np.ndarray
        3D VCG coordinates [n, 3].
    m_neighbors : int
        Number of VCG-space nearest neighbors.
    temporal_window : int
        Max sample lag for temporal contiguity link (default: 1).
    symmetric : bool
        Whether to enforce symmetry.

    Returns
    -------
    L_hybrid : scipy.sparse.csr_matrix
        Hybrid adjacency matrix [n, n].
    """
    L_vcg = construct_vcg_adjacency(vcg, m_neighbors=m_neighbors, symmetric=symmetric)
    n = len(vcg)

    if temporal_window <= 0 or n <= 1:
        return L_vcg

    rows = []
    cols = []
    for lag in range(1, temporal_window + 1):
        idx = np.arange(n - lag)
        rows.extend(idx)
        cols.extend(idx + lag)
        if symmetric:
            rows.extend(idx + lag)
            cols.extend(idx)

    data = np.ones(len(rows), dtype=int)
    L_time = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    L_hybrid = L_vcg.maximum(L_time)
    L_hybrid.setdiag(0)
    L_hybrid.eliminate_zeros()
    return L_hybrid
