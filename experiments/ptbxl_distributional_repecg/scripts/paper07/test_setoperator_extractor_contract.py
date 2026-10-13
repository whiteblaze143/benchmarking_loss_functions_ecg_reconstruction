#!/usr/bin/env python3
"""
Unit Test: SetOperator Canonical Extractor Numerical Agreement.
Verifies that freshly extracted E6 latent vectors from raw PTB-XL beats
using canonical response_fit.npz and response_atoms([x_q, dx_q/ds], atom_dim=2)
numerically match the stored canonical E6 latent vectors across 10 records.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

from repecg.paper07_operator.model import OperatorSetModel
from repecg.paper07_operator.measurements import response_atoms

# Paths
REPS_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations")
FIT_PATH = REPS_DIR / "response_fit.npz"
VAL_REPS_PATH = REPS_DIR / "representation_val.npz"
RECORDS_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper02_kernel_mean/development_select/records")
CKPT_PATH = Path("experiments/ptbxl_distributional_repecg/outputs/configuration_shift_evaluations/ptbxl/checkpoints/P07_ROBUSTNESS_TRAINED.pt")

EXPECTED_FIT_SHA = "83294d0d66b3e5a353a91a423955068ed7bd3934cfc983c9e0207ca58067b02d"
EXPECTED_CKPT_SHA = "1071470f58e63594341a7a5df589b3db50c56b86befa65cb47a0b1da2dccbf68"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_unit_test():
    print("======================================================================")
    print("SETOPERATOR EXTRACTOR CONTRACT UNIT TEST (10 PTB-XL RECORDS)")
    print("======================================================================")

    # 1. Assert Artifact Hashes
    fit_sha = compute_sha256(FIT_PATH)
    ckpt_sha = compute_sha256(CKPT_PATH)
    print(f"Response Fit Path: {FIT_PATH}")
    print(f"Response Fit SHA:  {fit_sha}")
    assert fit_sha == EXPECTED_FIT_SHA, f"Response fit SHA mismatch: {fit_sha} != {EXPECTED_FIT_SHA}"

    print(f"Checkpoint Path:   {CKPT_PATH}")
    print(f"Checkpoint SHA:    {ckpt_sha}")
    assert ckpt_sha == EXPECTED_CKPT_SHA, f"Checkpoint SHA mismatch: {ckpt_sha} != {EXPECTED_CKPT_SHA}"

    # 2. Load Response Fit Parameters
    fit = np.load(FIT_PATH)
    assert "landmarks" in fit, "Missing landmarks in fit"
    assert fit["landmarks"].shape == (128, 2), f"Expected landmarks shape (128, 2), got {fit['landmarks'].shape}"
    assert fit["whitening_mean"].shape == (2,), f"Expected whitening_mean shape (2,), got {fit['whitening_mean'].shape}"
    assert fit["whitening_components"].shape == (2, 2), f"Expected whitening_components shape (2, 2)"

    mean = fit["physical_mean"]
    std = fit["physical_std"]
    voltage_scale = float(fit["voltage_scale"])
    w_mean = torch.tensor(fit["whitening_mean"], dtype=torch.float32)
    w_comp = torch.tensor(fit["whitening_components"], dtype=torch.float32)
    w_scale = torch.tensor(fit["whitening_scales"], dtype=torch.float32)
    landmarks = torch.tensor(fit["landmarks"], dtype=torch.float32)
    inv_root = torch.tensor(fit["inverse_root"], dtype=torch.float32)
    c2 = float(fit["c2"])

    # 3. Load Model
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    model.load_state_dict(sd)
    model.eval()

    # 4. Load Validation Representations
    val_data = np.load(VAL_REPS_PATH)
    ecg_ids = val_data["ecg_ids"][:10]
    stored_responses = val_data["responses"][:10, 1]  # Lead II is index 1

    op_tensor = torch.zeros(1, 1, 8, dtype=torch.float32)
    op_tensor[0, 0, 1] = 1.0  # Lead II operator

    results = []
    max_all_diff = 0.0
    min_all_cos = 1.0

    q_lead2 = np.zeros(8, dtype=np.float64)
    q_lead2[1] = 1.0

    print(f"\nEvaluating 10 PTB-XL records against canonical representation cache:")
    for idx, ecg_id in enumerate(ecg_ids):
        rec_path = RECORDS_DIR / f"{ecg_id}.npz"
        assert rec_path.exists(), f"Record file not found: {rec_path}"
        rec = np.load(rec_path)
        beats = np.asarray(rec["beats"], dtype=np.float64) * std + mean

        # Canonical response_atoms: atom_dim MUST be 2
        atoms = response_atoms(beats, q_lead2, voltage_scale)
        assert atoms.ndim == 3 and atoms.shape[-1] == 2, f"Atom shape invalid: {atoms.shape}"

        # Nyström Phase Means
        cells = atoms.reshape(len(atoms), 16, 16, 2).transpose(1, 0, 2, 3).reshape(16, -1, 2)
        values = torch.as_tensor(cells, dtype=torch.float32)
        with torch.inference_mode():
            white = (values - w_mean) @ w_comp * w_scale
            features = torch.rsqrt(
                torch.cdist(white, landmarks.expand(16, -1, -1)).square() + c2
            ) @ inv_root
            fresh_resp = features.mean(dim=1).numpy().astype(np.float32)

            resp_stored_t = torch.tensor(stored_responses[idx]).unsqueeze(0).unsqueeze(0)
            resp_fresh_t = torch.tensor(fresh_resp).unsqueeze(0).unsqueeze(0)

            e6_stored = model.encode_context(op_tensor, resp_stored_t).squeeze(0).numpy()
            e6_fresh = model.encode_context(op_tensor, resp_fresh_t).squeeze(0).numpy()

        diff = np.abs(e6_fresh - e6_stored)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        cos_sim = float(np.dot(e6_fresh, e6_stored) / (np.linalg.norm(e6_fresh) * np.linalg.norm(e6_stored)))

        max_all_diff = max(max_all_diff, max_diff)
        min_all_cos = min(min_all_cos, cos_sim)

        # Assert tolerances per record
        assert max_diff < 0.002, f"Record {ecg_id}: max diff {max_diff:.6f} exceeded tolerance 0.002"
        assert cos_sim > 0.9999, f"Record {ecg_id}: cosine similarity {cos_sim:.6f} below 0.9999"

        print(f"  Record [{idx+1}/10] ecg_id={ecg_id:5d}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f} [PASS]")
        results.append({
            "record_index": idx,
            "ecg_id": int(ecg_id),
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "cosine_similarity": cos_sim,
            "status": "PASS"
        })

    summary = {
        "status": "PASS",
        "records_tested": len(results),
        "max_overall_abs_diff": max_all_diff,
        "min_overall_cosine_sim": min_all_cos,
        "response_fit_sha256": fit_sha,
        "checkpoint_sha256": ckpt_sha,
        "atom_dim": 2,
        "landmarks": 128,
        "tolerance_max_diff": 0.002,
        "tolerance_min_cos_sim": 0.9999,
        "per_record_results": results
    }

    out_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/setoperator_unit_test_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n======================================================================")
    print(f"ALL 10 RECORDS PASSED! Max Diff: {max_all_diff:.6f} | Min Cosine Sim: {min_all_cos:.8f}")
    print(f"Saved audit to: {out_path}")
    print("======================================================================")


if __name__ == "__main__":
    run_unit_test()
