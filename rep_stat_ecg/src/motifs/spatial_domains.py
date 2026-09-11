"""Spatial Discretization, Voxel Eligibility, and Constrained HAC Domain Discovery.

Implements Stage 1 of Q-VCG / RepSpat:
- Training-only coordinate standardizer: mu and MAD computed on Folds 1-7 only.
- 3-D occupancy lattice (24^3) in standardized coordinates.
- Voxel eligibility filter: n_sample >= n_sample_min, n_patient >= n_patient_min.
- 26-neighbor 3D spatial adjacency graph.
- Summary voxel attribute: a_c = [median(s), median(rho), median(kappa)].
- Spatially constrained Agglomerative Hierarchical Clustering (CAHC) into M contiguous domains G_1, ..., G_M.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering


class CoordinateStandardizer:
    """Robust 3D coordinate standardization using training-only median and MAD."""

    def __init__(self):
        self.median: np.ndarray | None = None  # [3]
        self.mad: np.ndarray | None = None     # [3]
        self.is_fitted: bool = False

    def fit(self, v: np.ndarray) -> CoordinateStandardizer:
        """Fits median and MAD on [N, 3] training VCG points."""
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"Expected v of shape [N, 3], got {v.shape}")
        self.median = np.median(v, axis=0)
        deviations = np.abs(v - self.median)
        self.mad = np.median(deviations, axis=0)
        # Ensure positive scale
        self.mad = np.maximum(self.mad, 1e-4)
        self.is_fitted = True
        return self

    def transform(self, v: np.ndarray) -> np.ndarray:
        """Standardizes [..., 3] points: v_tilde = (v - median) / (1.4826 * mad)."""
        if not self.is_fitted or self.median is None or self.mad is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        scale = 1.4826 * self.mad
        return (v - self.median) / scale

    def inverse_transform(self, v_tilde: np.ndarray) -> np.ndarray:
        """Restores physical coordinates from standardized points."""
        if not self.is_fitted or self.median is None or self.mad is None:
            raise RuntimeError("Standardizer must be fitted before inverse_transform")
        scale = 1.4826 * self.mad
        return (v_tilde * scale) + self.median


class VoxelGrid3D:
    """3D Occupancy Lattice with 26-neighbor adjacency."""

    def __init__(
        self,
        grid_dim: int = 24,
        coord_min: float = -3.0,
        coord_max: float = 3.0,
        n_sample_min: int = 30,
        n_patient_min: int = 5,
    ):
        self.grid_dim = grid_dim
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.n_sample_min = n_sample_min
        self.n_patient_min = n_patient_min

        self.bin_edges = np.linspace(coord_min, coord_max, grid_dim + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Set of eligible voxel linear indices
        self.eligible_voxels: set[int] = set()
        self.voxel_to_domain: dict[int, int] = {}
        self.domain_to_voxels: dict[int, list[int]] = {}

    def point_to_voxel_index(self, v_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Maps standardized 3D points [N, 3] to voxel linear indices.

        Returns:
            voxel_idx: [N] array of 1D linear voxel IDs (or -1 if out of grid bounds).
            in_bounds_mask: [N] boolean mask indicating valid points.
        """
        N = len(v_tilde)
        # Check bounds
        in_bounds = (
            (v_tilde[:, 0] >= self.coord_min) & (v_tilde[:, 0] <= self.coord_max) &
            (v_tilde[:, 1] >= self.coord_min) & (v_tilde[:, 1] <= self.coord_max) &
            (v_tilde[:, 2] >= self.coord_min) & (v_tilde[:, 2] <= self.coord_max)
        )

        ix = np.digitize(v_tilde[:, 0], self.bin_edges) - 1
        iy = np.digitize(v_tilde[:, 1], self.bin_edges) - 1
        iz = np.digitize(v_tilde[:, 2], self.bin_edges) - 1

        # Clip boundary exact equality to grid_dim - 1
        ix = np.clip(ix, 0, self.grid_dim - 1)
        iy = np.clip(iy, 0, self.grid_dim - 1)
        iz = np.clip(iz, 0, self.grid_dim - 1)

        linear_idx = ix * (self.grid_dim ** 2) + iy * self.grid_dim + iz
        voxel_idx = np.where(in_bounds, linear_idx, -1)
        return voxel_idx, in_bounds

    def linear_to_3d_index(self, linear_idx: int) -> tuple[int, int, int]:
        """Converts linear voxel ID to (ix, iy, iz)."""
        ix = linear_idx // (self.grid_dim ** 2)
        rem = linear_idx % (self.grid_dim ** 2)
        iy = rem // self.grid_dim
        iz = rem % self.grid_dim
        return int(ix), int(iy), int(iz)

    def get_26_neighbors(self, linear_idx: int) -> list[int]:
        """Returns valid 26-neighbors of a voxel within the 3D grid."""
        ix, iy, iz = self.linear_to_3d_index(linear_idx)
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                    if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim and 0 <= nz < self.grid_dim:
                        neighbors.append(nx * (self.grid_dim ** 2) + ny * self.grid_dim + nz)
        return neighbors

    def fit_spatial_domains(
        self,
        v_tilde: np.ndarray,
        xi: np.ndarray,
        patient_ids: list[str],
        n_domains: int = 64,
        linkage: str = "ward",
    ) -> dict[int, int]:
        """Discovers M contiguous spatial domains from training microstates.

        Args:
            v_tilde: [N, 3] standardized VCG positions.
            xi: [N, 3] intrinsic descriptors [s, rho, kappa].
            patient_ids: List of N patient identifiers.
            n_domains: Target number of contiguous spatial domains M.
            linkage: "ward", "complete", or "average".

        Returns:
            Mapping from eligible voxel_idx to domain_id in {0, ..., M-1}.
        """
        voxel_idx, in_bounds = self.point_to_voxel_index(v_tilde)

        # 1. Tally samples and distinct patients per voxel
        voxel_samples: dict[int, list[int]] = {}
        voxel_patients: dict[int, set[str]] = {}

        for i in range(len(v_tilde)):
            if not in_bounds[i]:
                continue
            vx = voxel_idx[i]
            if vx not in voxel_samples:
                voxel_samples[vx] = []
                voxel_patients[vx] = set()
            voxel_samples[vx].append(i)
            voxel_patients[vx].add(patient_ids[i])

        # 2. Filter eligible voxels
        self.eligible_voxels = {
            vx for vx, samples in voxel_samples.items()
            if len(samples) >= self.n_sample_min and len(voxel_patients[vx]) >= self.n_patient_min
        }

        if len(self.eligible_voxels) < n_domains:
            # If eligible voxels fewer than requested domains, relax thresholds gracefully
            self.eligible_voxels = {
                vx for vx, samples in voxel_samples.items()
                if len(samples) >= max(5, self.n_sample_min // 3)
            }

        sorted_voxels = sorted(list(self.eligible_voxels))
        V = len(sorted_voxels)
        if V == 0:
            raise RuntimeError("Zero eligible voxels found in training data!")

        actual_n_clusters = min(n_domains, V)

        # 3. Compute voxel attribute summary a_c = [median(s), median(rho), median(kappa)]
        voxel_to_pos = {vx: pos for pos, vx in enumerate(sorted_voxels)}
        X_voxel = np.zeros((V, 3), dtype=np.float32)

        for pos, vx in enumerate(sorted_voxels):
            sample_indices = voxel_samples[vx]
            xi_samples = xi[sample_indices]  # [N_vx, 3]
            X_voxel[pos] = np.median(xi_samples, axis=0)

        # Standardize voxel features
        feat_mean = np.mean(X_voxel, axis=0)
        feat_std = np.maximum(np.std(X_voxel, axis=0), 1e-4)
        X_voxel_std = (X_voxel - feat_mean) / feat_std

        # 4. Build 26-neighbor spatial connectivity sparse matrix
        row_ind, col_ind = [], []
        for pos, vx in enumerate(sorted_voxels):
            for nbr_vx in self.get_26_neighbors(vx):
                if nbr_vx in voxel_to_pos:
                    nbr_pos = voxel_to_pos[nbr_vx]
                    row_ind.append(pos)
                    col_ind.append(nbr_pos)

        data = np.ones(len(row_ind), dtype=np.int32)
        connectivity = sp.csr_matrix((data, (row_ind, col_ind)), shape=(V, V))

        # 5. Fit Constrained Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=actual_n_clusters,
            connectivity=connectivity,
            linkage=linkage,
        )
        labels = clustering.fit_predict(X_voxel_std)

        # Record mapping
        self.voxel_to_domain = {}
        self.domain_to_voxels = {d: [] for d in range(actual_n_clusters)}
        for pos, vx in enumerate(sorted_voxels):
            d = int(labels[pos])
            self.voxel_to_domain[vx] = d
            self.domain_to_voxels[d].append(vx)

        return self.voxel_to_domain
