"""Diagnostic audit of leave-one-patient-out Platt calibration across folds.

Inspects:
1. Per-fold logistic regression coefficients: slope (beta_1) and intercept (beta_0).
2. Within-patient AUROC for raw logits vs calibrated probabilities.
3. Pooled out-of-fold AUROC degradation mechanism.
4. Brier skill score against training prevalence baseline (0.1511).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import json

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import continuous_setoperator_models as csm
from run_stage1b_corrective_queue import prepare_training_dataset, prepare_validation_dataset, forward_dispatch
from torch.nn.attention import sdpa_kernel, SDPBackend

def audit_calibration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tensors = prepare_training_dataset(target_count=4096)
    norm_stats = train_tensors[6]
    informative_segments = [33, 86, 89, 90, 100, 118, 125, 126, 187, 237, 241, 261, 262, 263, 274]
    val_data = prepare_validation_dataset(informative_segments, norm_stats)

    patient_to_fis = {}
    for fi, seg in val_data.items():
        pid = seg["patient_id"]
        patient_to_fis.setdefault(pid, []).append(fi)
    unique_patients = sorted(list(patient_to_fis.keys()))

    target_models = ["A001", "A002", "A003", "A004", "D032-v1", "F053", "G061", "H071-v2", "I082-v1", "I082-v2"]
    audit_summary = {}

    for model_id in target_models:
        ckpt_path = Path("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/checkpoints") / f"{model_id}_best.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        mname = {
            "A001": "ContinuousSetOperatorA001",
            "A002": "MatchedCategoricalSetOperatorA002",
            "A003": "OperatorAblatedSetOperatorA003",
            "A004": "Continuous2DSetOperatorA004",
            "D032-v1": "DualMomentPhaseSetOperatorD032",
            "F053": "NoTransformerSetOperatorF053",
            "G061": "MultimodalRRSetOperatorG061",
            "H071-v2": "PerChannelRawResNetSetOperatorH071",
            "I082-v1": "ResNetResidualMLPSetOperatorI082",
            "I082-v2": "ResNetResidualFusionSetOperatorI082v2",
        }[model_id]

        model = getattr(csm, mname)().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Compute Pattern 2 logits
        seg_logits = {}
        for fi, seg in val_data.items():
            resp = seg["responses"].to(device)
            rr = seg["rr_features"].to(device)
            raw = seg["raw_waveforms"].to(device)
            B = len(resp)
            c_mask = torch.zeros(B, 2, dtype=torch.bool, device=device)
            ops = torch.zeros(B, 2, 8, device=device)
            ops[:, 0, 0] = 1.0; ops[:, 1, 1] = 1.0
            chunks = []
            for i in range(0, B, 512):
                with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                    sub = forward_dispatch(model, model_id, resp[i:i+512], ops[i:i+512], c_mask[i:i+512], rr[i:i+512], raw[i:i+512])
                    chunks.append(sub.cpu().numpy().astype(np.float64))
            seg_logits[fi] = np.concatenate(chunks)

        fold_records = []
        oof_raw_list = []
        oof_cal_list = []
        oof_tgt_list = []

        for p_held in unique_patients:
            tr_x, tr_y = [], []
            for p_other in unique_patients:
                if p_other == p_held:
                    continue
                for fi in patient_to_fis[p_other]:
                    tr_x.append(seg_logits[fi])
                    tr_y.append(val_data[fi]["targets"])
            tr_x = np.concatenate(tr_x).reshape(-1, 1)
            tr_y = np.concatenate(tr_y)

            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(tr_x, tr_y)
            slope = float(lr.coef_[0, 0])
            intercept = float(lr.intercept_[0])

            held_x = np.concatenate([seg_logits[fi] for fi in patient_to_fis[p_held]])
            held_y = np.concatenate([val_data[fi]["targets"] for fi in patient_to_fis[p_held]])
            held_cal = lr.predict_proba(held_x.reshape(-1, 1))[:, 1]

            oof_raw_list.append(held_x)
            oof_cal_list.append(held_cal)
            oof_tgt_list.append(held_y)

            has_both = len(np.unique(held_y)) > 1
            auc_raw = float(roc_auc_score(held_y, held_x)) if has_both else None
            auc_cal = float(roc_auc_score(held_y, held_cal)) if has_both else None

            fold_records.append({
                "held_out_patient": p_held,
                "slope_beta_1": slope,
                "intercept_beta_0": intercept,
                "within_patient_auc_raw": auc_raw,
                "within_patient_auc_cal": auc_cal,
                "af_prevalence": float(np.mean(held_y)),
                "num_windows": len(held_y),
            })

        pooled_raw = np.concatenate(oof_raw_list)
        pooled_cal = np.concatenate(oof_cal_list)
        pooled_tgt = np.concatenate(oof_tgt_list)
        pooled_auc_raw = float(roc_auc_score(pooled_tgt, pooled_raw))
        pooled_auc_cal = float(roc_auc_score(pooled_tgt, pooled_cal))
        brier_cal = float(brier_score_loss(pooled_tgt, pooled_cal))
        brier_prev = float(brier_score_loss(pooled_tgt, np.full_like(pooled_tgt, 0.1511, dtype=float)))
        bss = float(1.0 - brier_cal / brier_prev)

        valid_raw_aucs = [f["within_patient_auc_raw"] for f in fold_records if f["within_patient_auc_raw"] is not None]
        valid_cal_aucs = [f["within_patient_auc_cal"] for f in fold_records if f["within_patient_auc_cal"] is not None]

        audit_summary[model_id] = {
            "pooled_auc_raw": pooled_auc_raw,
            "pooled_auc_cal": pooled_auc_cal,
            "mean_within_patient_auc_raw": float(np.mean(valid_raw_aucs)) if valid_raw_aucs else None,
            "mean_within_patient_auc_cal": float(np.mean(valid_cal_aucs)) if valid_cal_aucs else None,
            "brier_calibrated": brier_cal,
            "brier_prevalence_baseline": brier_prev,
            "brier_skill_score": bss,
            "slopes": [f["slope_beta_1"] for f in fold_records],
            "intercepts": [f["intercept_beta_0"] for f in fold_records],
            "folds": fold_records,
        }

        print(f"\n[{model_id}] Pooled Raw AUC: {pooled_auc_raw:.4f} -> Calibrated AUC: {pooled_auc_cal:.4f}")
        print(f"  Within-patient Mean AUC: Raw = {np.mean(valid_raw_aucs):.4f} | Cal = {np.mean(valid_cal_aucs):.4f}")
        print(f"  Slopes (beta_1): min={min(f['slope_beta_1'] for f in fold_records):+.4f}, max={max(f['slope_beta_1'] for f in fold_records):+.4f}")
        print(f"  Intercepts (beta_0): min={min(f['intercept_beta_0'] for f in fold_records):+.4f}, max={max(f['intercept_beta_0'] for f in fold_records):+.4f}")
        print(f"  Brier: Cal = {brier_cal:.4f} vs Baseline = {brier_prev:.4f} | Brier Skill Score: {bss:+.4f}")

    out_file = Path("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/stage1c_calibration_diagnostics.json")
    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    local_out = SCRIPT_DIR / "stage1c_calibration_diagnostics.json"
    with open(local_out, "w") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n[Saved Diagnostics] -> {out_file}")

if __name__ == "__main__":
    audit_calibration()
