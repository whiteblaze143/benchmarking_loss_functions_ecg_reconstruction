from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from graphECG_author_code.graph import ECGGraphBuilder, LEAD_DEFINITIONS, LEAD_ORDER
from graphECG_author_code.model import GraphECG

logger = logging.getLogger("r025_graph_adapter")

# Known electrode pairs in the 8-lead coordinate convention [I, II, V1, V2, V3, V4, V5, V6]
# Lead I: LA - RA
# Lead II: LL - RA
# V1..V6: Vi - WCT
# V3 - V2: V3 - V2 (V2 -> V3)


class GraphECGAdapter:
    """Rigorous adapter for Stanford GraphECG.

    Constructs native GraphECG graphs for supported lead sets and single electrode-pair
    differences. Rejects dense/oblique linear combinations that do not correspond
    to a single physical electrode-pair difference, returning N/A.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.builder = ECGGraphBuilder()

    def build_graph_for_configuration(
        self,
        config_name: str,
        signal_8lead: np.ndarray,  # [T, 8] in [I, II, V1, V2, V3, V4, V5, V6]
    ) -> Optional[Data]:
        """Convert an 8-lead signal recording into a GraphECG Data object if representable.

        Returns:
            Data object, or None if the configuration contains dense/unsupported combinations.
        """
        # Full 12-lead reconstruction from 8 independent leads
        # I = signal[:, 0], II = signal[:, 1]
        # III = II - I
        # aVR = -(I + II)/2
        # aVL = (I - III)/2 = I - II/2
        # aVF = (II + III)/2 = II - I/2
        # V1..V6 = signal[:, 2:8]
        I = signal_8lead[:, 0]
        II = signal_8lead[:, 1]
        III = II - I
        aVR = -0.5 * (I + II)
        aVL = I - 0.5 * II
        aVF = II - 0.5 * I
        V1, V2, V3, V4, V5, V6 = [signal_8lead[:, i] for i in range(2, 8)]

        full_12 = {
            "I": I, "II": II, "III": III,
            "aVR": aVR, "aVL": aVL, "aVF": aVF,
            "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6,
        }

        if config_name == "Q8_full":
            # 8 independent leads present
            leads = {"I": I, "II": II, "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "S6_precordial":
            leads = {"V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "S6_limb":
            leads = {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "Derived_limb_only":
            leads = {"III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "Lead_III_single":
            leads = {"III": III}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "aVF_single":
            leads = {"aVF": aVF}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "S3_reduced":
            leads = {"I": I, "II": II, "V2": V2}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "S2_reduced":
            leads = {"I": I, "II": II}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "S1_smartwatch":
            leads = {"I": I}
            return self.builder.build_graph(leads, bidirectional=True)

        elif config_name == "S_icm":
            # Exact physical electrode pair V3 - V2: edge from V2 to V3
            icm_sig = V3 - V2
            edges = [("V2", "V3", icm_sig)]
            return self.builder.build_custom(edges, bidirectional=True)

        elif config_name == "S_icm_plus_leadI":
            icm_sig = V3 - V2
            edges = [("RA", "LA", I), ("V2", "V3", icm_sig)]
            return self.builder.build_custom(edges, bidirectional=True)

        elif config_name in {"Test_obliques_3", "All_16_leads"}:
            # These involve linear combinations of multiple electrodes that cannot
            # be uniquely represented by a single pair of physical electrode locations
            # without inventing unvalidated virtual nodes.
            return None

        else:
            return None


def load_graphecg_model(checkpoint_path: str, device: torch.device) -> GraphECG:
    """Load GraphECG checkpoint."""
    model = GraphECG(num_classes=5, tabular_dim=0)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
