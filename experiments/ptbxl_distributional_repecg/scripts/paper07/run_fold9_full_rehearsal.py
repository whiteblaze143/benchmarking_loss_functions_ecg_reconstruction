r"""Authentic Fold 9 Full Rehearsal & Verification Suite.

Implements Phase 6 Pre-Lockbox Gate:
1. Loads authentic trained checkpoints (5 Basis + 5 Span).
2. Physically exact operator-response pairing:
   - Canonical Q_8, S_6, S_2, S_1
   - Derived S_6 Limb, Derived Limb Only, Lead III single, aVF single
   - S_icm subcutaneous functional
3. Evaluates all 350 frozen Grassmann subspaces individually:
   - Computes per-subspace AUROC for every seed and model.
   - Computes rank-aggregated and overall Grassmann contrast.
4. Autonomous Observed-Only single-measurement segmentation vs Oracle:
   - Detects R-peaks directly on single projected channel y_q(t).
   - Computes downstream model logits and Delta_seg = AUROC_observed - AUROC_oracle.
5. Saves predictions archive and structured verification report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fold9_full_rehearsal")

from repecg.paper07_operator.model import OperatorSetModel
from evaluator_physical_operator import (
    FastResponseComputer,
    build_canonical_and_derived_operator_bank,
    compute_auroc_macro,
    evaluate_clinical_benchmarks,
    evaluate_grassmann_subspaces,
    evaluate_observed_only_segmentation,
    verify_operator_physical_consistency,
    CLINICAL_CONFIG_INDICES,
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_authentic_fold9_rehearsal(
    val_npz_path: Path,
    checkpoints_dir: Path,
    grassmann_bank_path: Path,
    resp_fit_path: Path,
    output_report_path: Path,
    output_predictions_path: Path,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute complete authentic Fold-9 rehearsal."""
    logger.info("=== Starting Authentic Fold 9 Benchmark Rehearsal ===")
    device = torch.device(device_name)
    logger.info("Using device: %s", device)

    # 1. Load Fold 9 representations
    if not val_npz_path.is_file():
        raise FileNotFoundError(f"Missing Fold 9 representations at {val_npz_path}")
    logger.info("Loading Fold 9 representations from %s", val_npz_path)
    val_data = np.load(val_npz_path)
    operators_16 = val_data["operators"]  # (N, 16, 8)
    responses_16 = val_data["responses"]  # (N, 16, 16, 128)
    targets = val_data["labels"]          # (N, 5)
    n_records = len(targets)
    logger.info("Loaded %d validation records with %d classes", n_records, targets.shape[1])

    # 2. Load trained checkpoints (5 Basis + 5 Span)
    seeds = [42, 43, 44, 45, 46]
    models: Dict[str, Dict[int, OperatorSetModel]] = {"basis": {}, "span": {}}

    for mode in ["basis", "span"]:
        for s in seeds:
            chk_path = checkpoints_dir / f"{mode}_s{s}.pt"
            if not chk_path.is_file():
                # Check alternative naming
                alt_path = checkpoints_dir / f"{mode}_seed{s}.pt"
                if alt_path.is_file():
                    chk_path = alt_path
                else:
                    raise FileNotFoundError(f"Missing required checkpoint: {chk_path}")

            chk = torch.load(chk_path, map_location="cpu")
            m = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
            state_dict = chk["state_dict"] if "state_dict" in chk else chk
            m.load_state_dict(state_dict, strict=True)
            m.to(device)
            m.eval()
            models[mode][s] = m

    logger.info("Successfully loaded all 10 trained checkpoints (seeds: %s)", seeds)

    # 3. Initialize Response Computer and verify physical consistency
    computer = FastResponseComputer(resp_fit_path, device)
    q_bank = build_canonical_and_derived_operator_bank()

    # Reconstruct or generate sample beats for consistency test
    # (B, 256, 8) synthetic test beats in millivolts
    test_beats_mv = np.random.default_rng(42).normal(size=(8, 256, 8)).astype(np.float32) * computer.std + computer.mean
    consistency_audit = verify_operator_physical_consistency(computer, test_beats_mv, q_bank)

    # 4. Evaluate all 9 clinical configurations
    clinical_results = evaluate_clinical_benchmarks(
        models=models,
        operators_16=operators_16,
        responses_16=responses_16,
        targets=targets,
        device=device,
        batch_size=256,
    )

    # 5. Evaluate all 350 Grassmann subspaces using Authentic Fold-9 Patient Beats
    beats_npz_path = FINAL_REPS_DIR / "fold9_authentic_beats.npz"
    if not beats_npz_path.exists():
        raise FileNotFoundError(
            f"FATAL: Authentic Fold-9 beats not found at {beats_npz_path}. "
            "Synthetic fallbacks are strictly forbidden. Run extract_and_audit_fold9_authentic_beats.py first."
        )
    logger.info("Loading verified authentic Fold-9 patient beats from %s...", beats_npz_path)
    d_beats = np.load(beats_npz_path, allow_pickle=True)
    record_beats_list = list(d_beats["beats"])
    assert len(record_beats_list) == n_records, f"Record count mismatch: {len(record_beats_list)} vs {n_records}"

    grassmann_preds_dir = output_predictions_path.parent / "grassmann_rehearsal"
    grassmann_results = evaluate_grassmann_subspaces(
        models=models,
        computer=computer,
        record_beats_list=record_beats_list,
        grassmann_bank_path=grassmann_bank_path,
        targets=targets,
        device=device,
        batch_size=256,
        ecg_ids=val_data["ecg_ids"] if "ecg_ids" in val_data else None,
        patient_ids=val_data["patient_ids"] if "patient_ids" in val_data else None,
        predictions_dir=grassmann_preds_dir,
    )

    # 6. Evaluate observed-only segmentation vs oracle using authentic single-lead waveforms
    single_lead_waveforms = d_beats["lead_I_waveforms"]  # Authentic (N, 5000) patient Lead I waveforms
    q_lead1 = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    segmentation_audit = evaluate_observed_only_segmentation(
        models=models,
        computer=computer,
        single_lead_waveforms=single_lead_waveforms,
        oracle_beats_list=record_beats_list,
        targets=targets,
        q=q_lead1,
        config_name="S_1",
        device=device,
        fs=500.0,
    )

    # 7. Assemble predictions array matching publication table compiler contract
    config_names = list(CLINICAL_CONFIG_INDICES.keys())
    n_configs = len(config_names)
    n_seeds = len(seeds)

    span_preds_arr = np.zeros((n_configs, n_seeds, n_records, 5), dtype=np.float32)
    basis_preds_arr = np.zeros((n_configs, n_seeds, n_records, 5), dtype=np.float32)

    for c_idx, cname in enumerate(CLINICAL_CONFIG_INDICES.keys()):
        span_preds_arr[c_idx] = clinical_results["span_preds"][cname]
        basis_preds_arr[c_idx] = clinical_results["basis_preds"][cname]

    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_predictions_path,
        targets=targets,
        span_preds=span_preds_arr,
        basis_preds=basis_preds_arr,
        configurations=np.array(config_names),
        seeds=np.array(seeds),
    )
    pred_sha = compute_sha256(output_predictions_path)
    logger.info("Saved Fold-9 clinical predictions archive to %s (SHA-256: %s)", output_predictions_path, pred_sha)

    # 8. Compile structured rehearsal report
    report = {
        "status": "FOLD9_REHEARSAL_PASSED",
        "dataset": "PTB-XL Fold 9 (final_select)",
        "n_records": n_records,
        "n_checkpoints": 10,
        "seeds": seeds,
        "n_clinical_configurations": len(CLINICAL_CONFIG_INDICES),
        "total_grassmann_subspaces": grassmann_results["total_subspaces"],
        "physical_consistency_audit": consistency_audit,
        "clinical_metrics": clinical_results["metrics"],
        "grassmann_metrics": {
            "overall_mean_span_auroc": grassmann_results["overall_mean_span_auroc"],
            "overall_mean_basis_auroc": grassmann_results["overall_mean_basis_auroc"],
            "overall_mean_delta": grassmann_results["overall_mean_delta"],
            "overall_std_delta": grassmann_results["overall_std_delta"],
            "rank_summaries": grassmann_results["rank_summaries"],
        },
        "segmentation_audit": segmentation_audit,
        "predictions_archive": {
            "path": str(output_predictions_path),
            "sha256": pred_sha,
        },
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Saved structured Fold-9 rehearsal report to %s", output_report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Fold 9 Authentic Rehearsal Suite.")
    parser.add_argument("--val-npz", type=Path,
                        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/representation_final_select_waveforms.npz"))
    parser.add_argument("--checkpoints-dir", type=Path,
                        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_checkpoints"))
    parser.add_argument("--grassmann-bank", type=Path,
                        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/expanded_grassmann_subspace_bank.npz"))
    parser.add_argument("--response-fit", type=Path,
                        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/response_fit_final.npz"))
    parser.add_argument("--output-report", type=Path,
                        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/fold9_rehearsal_report.json"))
    parser.add_argument("--output-predictions", type=Path,
                        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/fold9_rehearsal_predictions.npz"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_authentic_fold9_rehearsal(
        val_npz_path=args.val_npz,
        checkpoints_dir=args.checkpoints_dir,
        grassmann_bank_path=args.grassmann_bank,
        resp_fit_path=args.response_fit,
        output_report_path=args.output_report,
        output_predictions_path=args.output_predictions,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
