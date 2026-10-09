import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(EXP_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXP_DIR / "src"))

import numpy as np
import pytest
import torch

from scripts.paper07.r025_graph_adapter import GraphECGAdapter
from scripts.paper07.r025_csfm_adapter import (
    CSFMAdapter,
    ADMITTED_CSFM_BASE_SHA256,
)


def test_graphecg_adapter_standard_and_custom_configurations():
    device = torch.device("cpu")
    adapter = GraphECGAdapter(device=device)

    T = 1000
    signal_8lead = np.random.randn(T, 8).astype(np.float32)

    # 1. Standard Q8 full
    data_q8 = adapter.build_graph_for_configuration("Q8_full", signal_8lead)
    assert data_q8 is not None
    assert data_q8.edge_index.shape[0] == 2
    # 8 leads bidirectional -> 16 directed edges
    assert data_q8.edge_index.shape[1] == 16

    # 2. S6 precordial
    data_prec = adapter.build_graph_for_configuration("S6_precordial", signal_8lead)
    assert data_prec is not None
    assert data_prec.edge_index.shape[1] == 12

    # 3. S1 smartwatch (Lead I only)
    data_s1 = adapter.build_graph_for_configuration("S1_smartwatch", signal_8lead)
    assert data_s1 is not None
    assert data_s1.edge_index.shape[1] == 2

    # 4. S_icm: V3 - V2 (custom electrode pair)
    data_icm = adapter.build_graph_for_configuration("S_icm", signal_8lead)
    assert data_icm is not None
    assert data_icm.edge_index.shape[1] == 2

    # 5. Dense multi-electrode linear combinations must be rejected with None (N/A)
    data_oblique = adapter.build_graph_for_configuration("Test_obliques_3", signal_8lead)
    assert data_oblique is None, "GraphECG must not invent virtual nodes for arbitrary oblique combinations"

    data_all16 = adapter.build_graph_for_configuration("All_16_leads", signal_8lead)
    assert data_all16 is None


def test_csfm_adapter_admission_and_rejection():
    device = torch.device("cpu")
    repo = Path("external/Cardiac-Sensing-FM")
    ckpt = repo / "pretrained/CSFM_base.pth"

    adapter = CSFMAdapter(csfm_repo_path=repo, checkpoint_path=ckpt, device=device)
    assert adapter.admission_status == "ADMITTED", f"Expected ADMITTED, got {adapter.admission_status}"

    # Verify rejection of arbitrary non-standard channels
    bogus_dict = {"oblique_vector_1": np.random.randn(1000).astype(np.float32)}
    res = adapter.extract_features(bogus_dict)
    assert res is None, "CSFM must reject non-standard channel names with None"
