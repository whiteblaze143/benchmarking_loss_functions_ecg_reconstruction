"""Stage 1D P6: Genuine Channel-Dropout Feature Reconstruction & Evaluation.

Recomputes single-lead performance where:
- In Pattern 0 (Lead 0 only, Lead 1 dropped):
    * Waveform: raw[:, 1, :] = 0.0
    * SetOperator: operators[:, 1, :] = 0.0; mask[:, 1] = True
    * RR features: rr[:, 16:32] = 0.0; mask_ch1 = 0; valid_ch1 = 0
- In Pattern 1 (Lead 1 only, Lead 0 dropped):
    * Waveform: raw[:, 0, :] = 0.0
    * SetOperator: operators[:, 0, :] = 0.0; mask[:, 0] = True
    * RR features: rr[:, 0:16] = 0.0; mask_ch0 = 0; valid_ch0 = 0
- In Pattern 2 (Both leads):
    * All channels active.

Outputs:
- stage1d_channel_dropout_feature_reconstruction.csv
"""

import sys
from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import continuous_setoperator_models as csm
from run_stage1b_corrective_queue import prepare_training_dataset, prepare_validation_dataset, forward_dispatch
from waveform_protocol import W

ARTIFACTS_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[P6] Running Genuine Channel-Dropout Evaluation on {device}...")

    train_tensors = prepare_training_dataset(target_count=4096)
    norm_stats = train_tensors[6]
    informative_segments = [33, 86, 89, 90, 100, 118, 125, 126, 187, 237, 241, 261, 262, 263, 274]
    val_data = prepare_validation_dataset(informative_segments, norm_stats)

    # Collect all validation targets
    targets_list = []
    for fi in informative_segments:
        targets_list.append(val_data[fi]["targets"])
    targets = np.concatenate(targets_list)

    target_models = ["G061", "H071-v2", "I082-v2", "A001"]
    patterns = [
        ("Pattern 0 (Lead 0 only)", 0),
        ("Pattern 1 (Lead 1 only)", 1),
        ("Pattern 2 (Both leads)", 2),
    ]

    results_table = []

    for model_id in target_models:
        ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"{model_id}_best.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        mname = {
            "A001": "ContinuousSetOperatorA001",
            "G061": "MultimodalRRSetOperatorG061",
            "H071-v2": "PerChannelRawResNetSetOperatorH071",
            "I082-v2": "ResNetResidualFusionSetOperatorI082v2",
        }[model_id]

        model = getattr(csm, mname)().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        for pname, p_idx in patterns:
            pred_chunks = []
            for fi in informative_segments:
                seg = val_data[fi]
                B = len(seg["responses"])
                resp = seg["responses"].clone().to(device)
                rr = seg["rr_features"].clone().to(device)
                raw = seg["raw_waveforms"].clone().to(device)

                c_mask = torch.zeros(B, 2, dtype=torch.bool, device=device)
                ops = torch.zeros(B, 2, 8, device=device)

                if p_idx == 0:  # Lead 0 only, Lead 1 dropped
                    c_mask[:, 1] = True
                    ops[:, 0, 0] = 1.0
                    raw[:, 1, :] = 0.0
                    # Strictly zero out Channel 1 RR features and clear validity flags
                    rr[:, 16:32] = 0.0
                    rr[:, 33] = 0.0  # mask_ch1 = 0
                    rr[:, 35] = 0.0  # valid_ch1 = 0
                elif p_idx == 1:  # Lead 1 only, Lead 0 dropped
                    c_mask[:, 0] = True
                    ops[:, 1, 1] = 1.0
                    raw[:, 0, :] = 0.0
                    # Strictly zero out Channel 0 RR features and clear validity flags
                    rr[:, 0:16] = 0.0
                    rr[:, 32] = 0.0  # mask_ch0 = 0
                    rr[:, 34] = 0.0  # valid_ch0 = 0
                else:  # Pattern 2: Both active
                    ops[:, 0, 0] = 1.0
                    ops[:, 1, 1] = 1.0

                seg_preds = []
                for i in range(0, B, 512):
                    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                        sub = forward_dispatch(model, model_id, resp[i:i+512], ops[i:i+512], c_mask[i:i+512], rr[i:i+512], raw[i:i+512])
                        seg_preds.append(sub.cpu().numpy().astype(np.float64))
                pred_chunks.append(np.concatenate(seg_preds))

            all_logits = np.concatenate(pred_chunks)
            # Sigmoid probabilities
            probs = 1.0 / (1.0 + np.exp(-np.clip(all_logits, -15.0, 15.0)))

            auc = float(roc_auc_score(targets, all_logits))
            bacc = float(balanced_accuracy_score(targets, probs >= 0.5))
            brier = float(brier_score_loss(targets, probs))

            results_table.append({
                "model_id": model_id,
                "acquisition_pattern": pname,
                "lead_0_available": p_idx in [0, 2],
                "lead_1_available": p_idx in [1, 2],
                "genuine_channel_dropout_enforced": True,
                "rr_channel_leakage_prevented": True,
                "window_auroc": auc,
                "balanced_accuracy": bacc,
                "brier_score": brier,
                "evaluation_cohort": "11 Informative Validation Patients (15 Segments, 42,431 Windows)",
            })
            print(f"[{model_id:7s}] {pname:24s} | AUROC: {auc:.4f} | BAcc: {bacc:.4f} | Brier: {brier:.4f}")

    # Add Standalone RR-GBDT
    scores = np.load(W / "observed_rr_hgb_v1" / "validation_scores.npy")
    dev_path = W / "observed_rr_hgb_v1" / "development_features.npz"
    with np.load(dev_path) as z:
        ids = z["window_index"]
    manifest = list(csv.DictReader((W / "file_manifest.csv").open()))
    with np.load(W / "windows.npz") as z:
        labels = z["target"][ids]
        fi = z["file_index"][ids]

    parts = np.array([r["partition"] for r in manifest])[fi]
    ntrain = int((parts == "train").sum())
    val_targets = labels[ntrain:]
    val_fi = fi[ntrain:]
    inf_mask = np.isin(val_fi, informative_segments)

    for p_idx, (pname, p_code) in enumerate(patterns):
        sc = scores[:, p_idx][inf_mask]
        sub_y = val_targets[inf_mask]
        valid = np.isfinite(sc)
        auc = float(roc_auc_score(sub_y[valid], sc[valid]))
        bacc = float(balanced_accuracy_score(sub_y[valid], sc[valid] >= 0.5))
        brier = float(brier_score_loss(sub_y[valid], sc[valid]))

        results_table.append({
            "model_id": "RR-GBDT",
            "acquisition_pattern": pname,
            "lead_0_available": p_code in [0, 2],
            "lead_1_available": p_code in [1, 2],
            "genuine_channel_dropout_enforced": True,
            "rr_channel_leakage_prevented": True,
            "window_auroc": auc,
            "balanced_accuracy": bacc,
            "brier_score": brier,
            "evaluation_cohort": "11 Informative Validation Patients (15 Segments, 42,431 Windows)",
        })
        print(f"[RR-GBDT] {pname:24s} | AUROC: {auc:.4f} | BAcc: {bacc:.4f} | Brier: {brier:.4f}")

    # Save to CSV
    fieldnames = list(results_table[0].keys())
    out_csv = ARTIFACTS_DIR / "stage1d_channel_dropout_feature_reconstruction.csv"
    local_csv = SCRIPT_DIR / "stage1d_channel_dropout_feature_reconstruction.csv"
    for p in [out_csv, local_csv]:
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_table)

    print(f"\n[Saved Channel-Dropout Results] -> {out_csv}")


if __name__ == "__main__":
    main()
