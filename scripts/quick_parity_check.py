#!/usr/bin/env python3
"""Quick Numerical Parity Check for SetOperator & G1 on 200 Validation Records."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
sys.path.insert(0, str(DEV_ROOT / "experiments/echonext_operator_benchmark"))
sys.path.insert(0, str(DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"))

from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH, CANONICAL_8_LEADS
from train_setoperator_echonext import EchoNextSetOperatorDataset, collate_set_fn, build_setoperator_echonext, DATA_DIR
from evaluate_ablation_comparison import EchoNextPreloadedCohort
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext

BASELINE_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
G1_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/g1_setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")


def check_model(name: str, ckpt_path: Path, is_g1: bool = False, n_records: int = 200):
    device = torch.device("cpu")
    chk = torch.load(ckpt_path, map_location="cpu")
    if is_g1:
        model = AblationSetOperatorEchoNext(geometry_mode="G1", classes=1).to(device)
    else:
        model = build_setoperator_echonext("SO-ECG+Tab-Native").to(device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values[:n_records]
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")[:n_records]
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")[:n_records]

    # Pipeline A: Training DataLoader
    ds_A = EchoNextSetOperatorDataset(val_w, val_t, y_val, regime="SO-12", representation="phase_kme", seed=42, is_train=False)
    loader_A = DataLoader(ds_A, batch_size=64, shuffle=False, collate_fn=collate_set_fn)
    logits_A, probs_A = [], []
    with torch.no_grad():
        for b in loader_A:
            out = model(operators=b["operators"].to(device), responses=b["responses"].to(device), tabular=b["tabular"].to(device), mask=b["mask"].to(device))
            l = out["logits"].squeeze(-1)
            logits_A.extend(l.numpy().tolist())
            probs_A.extend(torch.sigmoid(l).numpy().tolist())

    # Pipeline B: Standalone Evaluator
    extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device="cpu")
    cohort_B = EchoNextPreloadedCohort(val_w, val_t, y_val, extractor, limit=n_records)
    ops = np.eye(8, dtype=np.float32)
    ops_t = torch.from_numpy(ops).unsqueeze(0)
    logits_B, probs_B = [], []
    with torch.no_grad():
        for i in range(0, n_records, 64):
            b_sz = min(64, n_records - i)
            batch_resps = [extractor.compute_phase_kme(cohort_B.beats[i + j], torch.from_numpy(ops)) for j in range(b_sz)]
            t_resps = torch.stack(batch_resps)
            t_tabs = cohort_B.tabular[i : i + b_sz]
            t_ops = ops_t.expand(b_sz, -1, -1)
            out = model(operators=t_ops, responses=t_resps, tabular=t_tabs, mask=None)
            l = out["logits"].squeeze(-1)
            logits_B.extend(l.numpy().tolist())
            probs_B.extend(torch.sigmoid(l).numpy().tolist())

    l_A = np.array(logits_A)
    l_B = np.array(logits_B)
    p_A = np.array(probs_A)
    p_B = np.array(probs_B)

    max_logit_diff = np.max(np.abs(l_A - l_B))
    mean_logit_diff = np.mean(np.abs(l_A - l_B))
    max_prob_diff = np.max(np.abs(p_A - p_B))
    mean_prob_diff = np.mean(np.abs(p_A - p_B))

    print(f"=== {name} Parity on {n_records} Records ===")
    print(f"Max Logit Diff:       {max_logit_diff:.3e}")
    print(f"Mean Logit Diff:      {mean_logit_diff:.3e}")
    print(f"Max Probability Diff: {max_prob_diff:.3e}")
    print(f"Mean Probability Diff:{mean_prob_diff:.3e}")
    print(f"Are probabilities identical (atol=1e-5)? {np.allclose(p_A, p_B, atol=1e-5)}")
    print(f"First 5 Pipeline A Probs: {p_A[:5]}")
    print(f"First 5 Pipeline B Probs: {p_B[:5]}")


def main():
    check_model("Original SetOperator SO-Span s42", BASELINE_CKPT, is_g1=False, n_records=200)
    check_model("G1 Geometry-Aware SetOperator s42", G1_CKPT, is_g1=True, n_records=200)


if __name__ == "__main__":
    main()
