"""Frozen VCG-to-Motif Tokenizer with Explicit Out-Of-Distribution (OOD) Handling.

Maps continuous physical VCG trajectory v(t) in R^3 to discrete motif sequence:
    q: V -> {R_0, ..., R_{K-1}, OOD}

Pipeline:
1. Standardize v using frozen training parameters (median, MAD).
2. Look up 3D voxel index on 24^3 grid.
3. If voxel outside grid or not in eligible training voxels -> OOD (index K).
4. Otherwise: voxel -> domain G_i -> motif R_k.
"""
from __future__ import annotations

import numpy as np
from .spatial_domains import CoordinateStandardizer, VoxelGrid3D
from .quotient import QuotientClosure


class QVCGTokenizer:
    """Discrete Tokenizer for 3D VCG Phase Space."""

    def __init__(
        self,
        standardizer: CoordinateStandardizer,
        grid: VoxelGrid3D,
        quotient: QuotientClosure,
    ):
        self.standardizer = standardizer
        self.grid = grid
        self.quotient = quotient
        self.num_motifs = quotient.num_motifs
        self.ood_token = self.num_motifs  # Token ID K reserved for OOD

    def tokenize(self, v_physical: np.ndarray) -> tuple[np.ndarray, float]:
        """Maps physical VCG trajectory [..., 3, T] to integer motif tokens [..., T].

        Args:
            v_physical: Array of shape [..., 3, T] in physical mV.

        Returns:
            tokens: Array of integer motif IDs [..., T] in {0, ..., K-1} or OOD (K).
            ood_rate: Fraction of microstates flagged as OOD.
        """
        if v_physical.ndim == 2:
            if v_physical.shape[0] != 3:
                raise ValueError(f"Expected shape [3, T], got {v_physical.shape}")
            T = v_physical.shape[1]
            v_flat = v_physical.T  # [T, 3]
            batch_shape = ()
        elif v_physical.ndim == 3:
            if v_physical.shape[1] != 3:
                raise ValueError(f"Expected shape [B, 3, T], got {v_physical.shape}")
            B, _, T = v_physical.shape
            v_flat = np.moveaxis(v_physical, 1, -1).reshape(-1, 3)
            batch_shape = (B,)
        else:
            raise ValueError(f"Expected 2D or 3D array with 3 coordinates, got {v_physical.shape}")

        # 1. Standardize
        v_tilde = self.standardizer.transform(v_flat)

        # 2. Voxel lookup
        voxel_idx, in_bounds = self.grid.point_to_voxel_index(v_tilde)

        # 3. Map to domain and then motif
        tokens_flat = np.full(len(v_flat), fill_value=self.ood_token, dtype=np.int32)

        for i in range(len(v_flat)):
            if not in_bounds[i]:
                continue
            vx = voxel_idx[i]
            if vx in self.grid.eligible_voxels and vx in self.grid.voxel_to_domain:
                d = self.grid.voxel_to_domain[vx]
                if d in self.quotient.domain_to_motif:
                    tokens_flat[i] = self.quotient.domain_to_motif[d]

        # Reshape back to [..., T]
        if batch_shape:
            tokens = tokens_flat.reshape(*batch_shape, T)
        else:
            tokens = tokens_flat.reshape(T)

        ood_count = np.sum(tokens_flat == self.ood_token)
        ood_rate = float(ood_count / max(len(tokens_flat), 1))

        return tokens, ood_rate

    def tokenize_domains(self, v_physical: np.ndarray) -> tuple[np.ndarray, float]:
        """Maps physical VCG trajectory to unmerged domain IDs in {0, ..., M-1} or OOD."""
        if v_physical.ndim == 2:
            if v_physical.shape[0] != 3:
                raise ValueError(f"Expected shape [3, T], got {v_physical.shape}")
            T = v_physical.shape[1]
            v_flat = v_physical.T
            batch_shape = ()
        elif v_physical.ndim == 3:
            if v_physical.shape[1] != 3:
                raise ValueError(f"Expected shape [B, 3, T], got {v_physical.shape}")
            B, _, T = v_physical.shape
            v_flat = np.moveaxis(v_physical, 1, -1).reshape(-1, 3)
            batch_shape = (B,)
        else:
            raise ValueError(f"Expected 2D or 3D array with 3 coordinates, got {v_physical.shape}")

        v_tilde = self.standardizer.transform(v_flat)
        voxel_idx, in_bounds = self.grid.point_to_voxel_index(v_tilde)

        ood_domain = self.grid.grid_dim ** 3  # out-of-domain flag
        domains_flat = np.full(len(v_flat), fill_value=ood_domain, dtype=np.int32)

        for i in range(len(v_flat)):
            if not in_bounds[i]:
                continue
            vx = voxel_idx[i]
            if vx in self.grid.eligible_voxels and vx in self.grid.voxel_to_domain:
                domains_flat[i] = self.grid.voxel_to_domain[vx]

        if batch_shape:
            domains = domains_flat.reshape(*batch_shape, T)
        else:
            domains = domains_flat.reshape(T)
        ood_rate = float(np.mean(domains_flat == ood_domain))
        return domains, ood_rate
