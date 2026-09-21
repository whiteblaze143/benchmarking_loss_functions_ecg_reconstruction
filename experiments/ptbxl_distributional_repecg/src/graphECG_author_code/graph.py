"""
ECG Graph Construction

Represents ECG as a directed graph where electrodes are nodes and leads are edges.
Each lead measures potential difference: x_ij(t) = phi_j(t) - phi_i(t)
"""

import math
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional


# Electrode positions derived from standard ECG lead geometry
# Positions computed to satisfy Einthoven's law and standard lead angles
ELECTRODE_POSITIONS = {
    'WCT': np.array([0.0, 0.0, 0.0], dtype=np.float32),
    'RA': np.array([-0.449, -0.596, 0.174], dtype=np.float32),
    'LA': np.array([0.551, -0.096, 0.174], dtype=np.float32),
    'LL': np.array([-0.101, 0.693, 0.674], dtype=np.float32),
    'V1': np.array([0.999, -0.017, 0.0], dtype=np.float32),
    'V2': np.array([0.996, 0.087, 0.0], dtype=np.float32),
    'V3': np.array([0.949, 0.259, 0.174], dtype=np.float32),
    'V4': np.array([0.813, 0.470, 0.342], dtype=np.float32),
    'V5': np.array([0.663, 0.663, 0.342], dtype=np.float32),
    'V6': np.array([0.500, 0.866, 0.0], dtype=np.float32),
    'mid_LA_LL': np.array([0.225, 0.298, 0.424], dtype=np.float32),
    'mid_RA_LL': np.array([-0.275, 0.048, 0.424], dtype=np.float32),
    'mid_RA_LA': np.array([0.051, -0.346, 0.174], dtype=np.float32),
}

ALL_ELECTRODES = list(ELECTRODE_POSITIONS.keys())

# Standard 12-lead definitions as (source, target) electrode pairs
LEAD_DEFINITIONS = {
    'I': ('RA', 'LA'),
    'II': ('RA', 'LL'),
    'III': ('LA', 'LL'),
    'aVR': ('mid_LA_LL', 'RA'),
    'aVL': ('mid_RA_LL', 'LA'),
    'aVF': ('mid_RA_LA', 'LL'),
    'V1': ('WCT', 'V1'),
    'V2': ('WCT', 'V2'),
    'V3': ('WCT', 'V3'),
    'V4': ('WCT', 'V4'),
    'V5': ('WCT', 'V5'),
    'V6': ('WCT', 'V6'),
}

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _direction_to_spherical(src_pos: np.ndarray, tgt_pos: np.ndarray) -> Tuple[float, float]:
    direction = tgt_pos - src_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return 0.0, 0.0
    d = direction / norm
    phi = math.acos(np.clip(d[2], -1.0, 1.0))
    theta = math.atan2(d[1], d[0])
    if theta < 0:
        theta += 2 * math.pi
    return theta, phi


class ECGGraphBuilder:
    """Builds PyG graphs from ECG signals."""

    def __init__(self):
        self.electrodes = ALL_ELECTRODES
        self.electrode_to_idx = {e: i for i, e in enumerate(self.electrodes)}
        self.positions = torch.tensor(
            np.array([ELECTRODE_POSITIONS[e] for e in self.electrodes]),
            dtype=torch.float32
        )

    def build_graph(
        self,
        signals: Dict[str, np.ndarray],
        bidirectional: bool = True,
    ) -> Data:
        """
        Build graph from observed lead signals.

        Args:
            signals: Dict mapping lead name to signal array, e.g. {'I': array, 'II': array}
            bidirectional: If True, include reverse edges with negated signals

        Returns:
            PyG Data object
        """
        edge_src, edge_tgt = [], []
        edge_attr_list, edge_spherical = [], []

        for lead_name, signal in signals.items():
            if lead_name not in LEAD_DEFINITIONS:
                raise ValueError(f"Unknown lead: {lead_name}")

            src, tgt = LEAD_DEFINITIONS[lead_name]
            src_idx = self.electrode_to_idx[src]
            tgt_idx = self.electrode_to_idx[tgt]

            signal = np.asarray(signal, dtype=np.float32)

            edge_src.append(src_idx)
            edge_tgt.append(tgt_idx)
            edge_attr_list.append(signal)
            theta, phi = _direction_to_spherical(
                ELECTRODE_POSITIONS[src], ELECTRODE_POSITIONS[tgt]
            )
            edge_spherical.append([theta, phi])

            if bidirectional:
                edge_src.append(tgt_idx)
                edge_tgt.append(src_idx)
                edge_attr_list.append(-signal)
                theta, phi = _direction_to_spherical(
                    ELECTRODE_POSITIONS[tgt], ELECTRODE_POSITIONS[src]
                )
                edge_spherical.append([theta, phi])

        return Data(
            x=self.positions.clone(),
            edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
            edge_attr=torch.tensor(np.stack(edge_attr_list), dtype=torch.float32),
            edge_spherical=torch.tensor(edge_spherical, dtype=torch.float32),
        )

    def build_from_array(
        self,
        ecg: np.ndarray,
        lead_indices: Optional[List[int]] = None,
        bidirectional: bool = True,
    ) -> Data:
        """
        Build graph from 12-lead array.

        Args:
            ecg: Array of shape (12, T) or (T, 12)
            lead_indices: Which leads to include (0-11). None = all 12.
            bidirectional: Include reverse edges
        """
        if ecg.shape[0] != 12:
            ecg = ecg.T
        if lead_indices is None:
            lead_indices = list(range(12))

        signals = {LEAD_ORDER[i]: ecg[i] for i in lead_indices}
        return self.build_graph(signals, bidirectional=bidirectional)

    def build_custom(self, edges, bidirectional: bool = True) -> Data:
        """
        Build a graph from arbitrary electrode-pair measurements.

        This is what makes non-standard leads natural: any measured potential
        difference between two known electrodes is just an edge. E.g. an ICM /
        wearable vector V3-V2 is a single edge from V2 to V3.

        Args:
            edges: list of (src_electrode, tgt_electrode, signal) tuples, where
                signal is the waveform of (phi_tgt - phi_src). Electrode names
                must be keys of ELECTRODE_POSITIONS.
            bidirectional: also add reverse edges carrying the negated signal.
        """
        edge_src, edge_tgt, edge_attr_list, edge_spherical = [], [], [], []
        for src, tgt, signal in edges:
            si, ti = self.electrode_to_idx[src], self.electrode_to_idx[tgt]
            signal = np.asarray(signal, dtype=np.float32)
            edge_src.append(si)
            edge_tgt.append(ti)
            edge_attr_list.append(signal)
            edge_spherical.append(list(_direction_to_spherical(
                ELECTRODE_POSITIONS[src], ELECTRODE_POSITIONS[tgt])))
            if bidirectional:
                edge_src.append(ti)
                edge_tgt.append(si)
                edge_attr_list.append(-signal)
                edge_spherical.append(list(_direction_to_spherical(
                    ELECTRODE_POSITIONS[tgt], ELECTRODE_POSITIONS[src])))
        return Data(
            x=self.positions.clone(),
            edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
            edge_attr=torch.tensor(np.stack(edge_attr_list), dtype=torch.float32),
            edge_spherical=torch.tensor(edge_spherical, dtype=torch.float32),
        )
