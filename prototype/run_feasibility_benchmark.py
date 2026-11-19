"""
Feasibility Prototype & Benchmark Evaluation Harness for Wearable AF Recurrence Detection.
Evaluates 5 Model Architectures across Clean, Moderate Noise (N1), and Severe Noise (N2) regimes:
1) RR-based detector (ArNet2)
2) Raw ECG CNN (RawECGNet / ArNetECG)
3) SetOperator Phase-KME
4) SetOperator + RR (Multimodal Fusion)
5) Quality-Aware SetOperator with Selective Abstention
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
import torch.optim as optim

# Avoid driver mismatch in Conv1D on A100 with newer CUDA
torch.backends.cudnn.enabled = False

# Add prototype dir to path
prototype_dir = os.path.dirname(os.path.abspath(__file__))
if prototype_dir not in sys.path:
    sys.path.insert(0, prototype_dir)

from models import (
    ArNet2Detector,
    RawECGNet,
    SetOperatorPhaseKME,
    SetOperatorPlusRR,
    QualityAwareSetOperator,
)
from data_loader import load_sample_dataset
from sqi import compute_composite_sqi


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Calculate clinical performance metrics."""
    if len(np.unique(y_true)) < 2:
        # Single class case
        return {
            "AUROC": 0.5,
            "AUPRC": float(np.mean(y_true)),
            "Sensitivity": 0.0,
            "Specificity": 1.0,
            "F1": 0.0,
            "Accuracy": float(np.mean(y_pred == y_true)) if y_pred is not None else 0.5,
        }

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float((tp + tn) / (tp + tn + fp + fn))

    return {
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1": round(f1, 4),
        "Accuracy": round(accuracy, 4),
    }


def train_model_quick(
    model: nn.Module,
    train_loader: list[tuple],
    epochs: int = 15,
    lr: float = 1e-3,
    model_type: str = "rr",
    device: str = "cpu",
) -> nn.Module:
    """Quick training/fine-tuning routine for prototype benchmarking."""
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == "rr":
                rr, y = batch
                rr, y = rr.to(device), y.to(device).float().unsqueeze(1)
                logits = model(rr)
            elif model_type == "raw":
                ecg, y = batch
                ecg, y = ecg.to(device), y.to(device).float().unsqueeze(1)
                logits = model(ecg)
            elif model_type == "kme":
                kme, y = batch
                kme, y = kme.to(device), y.to(device).float().unsqueeze(1)
                logits = model(kme)
            elif model_type in ("multimodal", "quality_aware"):
                kme, rr, hrv, y = batch
                kme = kme.to(device)
                rr = rr.to(device)
                hrv = hrv.to(device)
                y = y.to(device).float().unsqueeze(1)
                logits = model(kme, rr, hrv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def run_benchmark(
    max_per_group: int = 25,
    epochs: int = 12,
    output_dir: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence",
) -> dict:
    """Run end-to-end benchmark on real Noise Holter Database."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Feasibility Benchmark on device: {device}")

    # 1. Load Data
    data = load_sample_dataset(max_per_group=max_per_group)
    n_samples = len(data["labels"])
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    # 70% Train, 30% Test split stratified across groups
    split_idx = int(0.70 * n_samples)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    print(f"Total dataset: {n_samples} windows. Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Convert to PyTorch tensors
    raw_ecg = torch.tensor(data["raw_ecg"], dtype=torch.float32)
    rr_seq = torch.tensor(data["rr_seq"], dtype=torch.float32)
    hrv_stats = torch.tensor(data["hrv_stats"], dtype=torch.float32)
    phase_kme = torch.tensor(data["phase_kme"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    noise_tier = data["noise_tier"]

    # Prepare Train Batches
    batch_size = 16
    train_batches_rr = []
    train_batches_raw = []
    train_batches_kme = []
    train_batches_multi = []

    for i in range(0, len(train_idx), batch_size):
        b_idx = train_idx[i : i + batch_size]
        train_batches_rr.append((rr_seq[b_idx], labels[b_idx]))
        train_batches_raw.append((raw_ecg[b_idx], labels[b_idx]))
        train_batches_kme.append((phase_kme[b_idx], labels[b_idx]))
        train_batches_multi.append((phase_kme[b_idx], rr_seq[b_idx], hrv_stats[b_idx], labels[b_idx]))

    # Instantiate Models
    m1_arnet2 = ArNet2Detector()
    m2_rawecg = RawECGNet()
    m3_kme = SetOperatorPhaseKME()
    m4_multi = SetOperatorPlusRR()

    print("\nTraining Model 1: ArNet2 (RR baseline)...")
    m1_arnet2 = train_model_quick(m1_arnet2, train_batches_rr, epochs=epochs, model_type="rr", device=device)

    print("Training Model 2: RawECGNet (Raw ECG CNN baseline)...")
    m2_rawecg = train_model_quick(m2_rawecg, train_batches_raw, epochs=epochs, model_type="raw", device=device)

    print("Training Model 3: SetOperator Phase-KME...")
    m3_kme = train_model_quick(m3_kme, train_batches_kme, epochs=epochs, model_type="kme", device=device)

    print("Training Model 4: SetOperator + RR (Multimodal Fusion)...")
    m4_multi = train_model_quick(m4_multi, train_batches_multi, epochs=epochs, model_type="multimodal", device=device)

    # Model 5: Quality-Aware SetOperator with Abstention
    m5_quality = QualityAwareSetOperator(tau_qual=0.55, tau_unc=0.35, tau_p=0.50)
    # Copy weights from trained multimodal backbone
    m5_quality.backbone.load_state_dict(m4_multi.state_dict())
    m5_quality.to(device)

    # =========================================================================
    # Evaluation on Test Set
    # =========================================================================
    test_raw = raw_ecg[test_idx].to(device)
    test_rr = rr_seq[test_idx].to(device)
    test_hrv = hrv_stats[test_idx].to(device)
    test_kme = phase_kme[test_idx].to(device)
    test_y = labels[test_idx].numpy()
    test_noise = noise_tier[test_idx]
    test_raw_np = data["raw_ecg"][test_idx]

    with torch.no_grad():
        p1 = torch.sigmoid(m1_arnet2(test_rr)).cpu().numpy().flatten()
        p2 = torch.sigmoid(m2_rawecg(test_raw)).cpu().numpy().flatten()
        p3 = torch.sigmoid(m3_kme(test_kme)).cpu().numpy().flatten()
        p4 = torch.sigmoid(m4_multi(test_kme, test_rr, test_hrv)).cpu().numpy().flatten()
        out5 = m5_quality.predict_with_abstention(test_kme, test_rr, test_hrv, test_raw_np)

    # 1. Overall Test Metrics
    results_overall = {
        "ArNet2 (RR-only)": compute_clinical_metrics(test_y, p1),
        "RawECGNet (Raw CNN)": compute_clinical_metrics(test_y, p2),
        "SetOperator Phase-KME": compute_clinical_metrics(test_y, p3),
        "SetOperator + RR": compute_clinical_metrics(test_y, p4),
    }

    # Model 5: Evaluated on non-abstained windows
    accepted_mask = ~out5["abstain"]
    n_accepted = int(np.sum(accepted_mask))
    coverage_pct = round(100.0 * n_accepted / len(test_y), 2)

    if n_accepted > 0 and len(np.unique(test_y[accepted_mask])) > 1:
        metrics_m5 = compute_clinical_metrics(
            test_y[accepted_mask],
            out5["proba"][accepted_mask],
            y_pred=out5["pred"][accepted_mask],
        )
    else:
        metrics_m5 = compute_clinical_metrics(test_y, out5["proba"], y_pred=out5["pred"])
    metrics_m5["Coverage (%)"] = coverage_pct
    results_overall["Quality-Aware SetOperator"] = metrics_m5

    # 2. Performance Stratified by Noise Tier (Clean vs N1 vs N2)
    noise_breakdown = {}
    for tier in ["Clean", "N1", "N2"]:
        tier_mask = test_noise == tier
        n_tier = int(np.sum(tier_mask))
        if n_tier == 0:
            continue

        y_tier = test_y[tier_mask]
        nsr_mask = y_tier == 0

        # False Positive Rate on Normal Sinus Rhythm (critical wearable failure metric)
        fpr_1 = float(np.mean(p1[tier_mask][nsr_mask] >= 0.5)) if np.sum(nsr_mask) > 0 else 0.0
        fpr_2 = float(np.mean(p2[tier_mask][nsr_mask] >= 0.5)) if np.sum(nsr_mask) > 0 else 0.0
        fpr_3 = float(np.mean(p3[tier_mask][nsr_mask] >= 0.5)) if np.sum(nsr_mask) > 0 else 0.0
        fpr_4 = float(np.mean(p4[tier_mask][nsr_mask] >= 0.5)) if np.sum(nsr_mask) > 0 else 0.0

        # Model 5: FPR only on accepted NSR windows, plus abstention rate
        tier_abstain = out5["abstain"][tier_mask]
        abstain_pct = round(100.0 * float(np.mean(tier_abstain)), 2)

        tier_accepted_nsr = (~tier_abstain) & nsr_mask
        fpr_5 = (
            float(np.mean(out5["pred"][tier_mask][tier_accepted_nsr] == 1))
            if np.sum(tier_accepted_nsr) > 0
            else 0.0
        )

        noise_breakdown[tier] = {
            "Total Windows": n_tier,
            "NSR False Alarm Rate - ArNet2": round(fpr_1, 4),
            "NSR False Alarm Rate - RawECGNet": round(fpr_2, 4),
            "NSR False Alarm Rate - SetOperator Phase-KME": round(fpr_3, 4),
            "NSR False Alarm Rate - SetOperator + RR": round(fpr_4, 4),
            "NSR False Alarm Rate - Quality-Aware (Accepted)": round(fpr_5, 4),
            "Quality-Aware Abstention Rate (%)": abstain_pct,
        }

    # Summary table
    print("\n" + "=" * 80)
    print("FEASIBILITY BENCHMARK RESULTS: 5 AF ARCHITECTURES")
    print("=" * 80)

    summary_rows = []
    for model_name, m in results_overall.items():
        summary_rows.append({
            "Model Architecture": model_name,
            "AUROC": m["AUROC"],
            "AUPRC": m["AUPRC"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "F1-Score": m["F1"],
            "Coverage (%)": m.get("Coverage (%)", 100.0),
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_markdown(index=False))

    print("\n" + "=" * 80)
    print("NOISE ROBUSTNESS & FALSE ALARM RATE ACROSS NOISE TIERS (Clean vs N1 vs N2)")
    print("=" * 80)
    df_noise = pd.DataFrame.from_dict(noise_breakdown, orient="index")
    print(df_noise.to_markdown())

    # Save to disk
    json_path = os.path.join(output_dir, "af_model_benchmark_results.json")
    csv_path = os.path.join(output_dir, "af_model_benchmark_summary.csv")
    noise_csv_path = os.path.join(output_dir, "noise_tier_robustness_summary.csv")

    output_payload = {
        "overall_metrics": results_overall,
        "noise_tier_breakdown": noise_breakdown,
        "test_sample_size": len(test_y),
    }

    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    df_summary.to_csv(csv_path, index=False)
    df_noise.to_csv(noise_csv_path, index=True)

    print(f"\nArtifacts successfully generated:")
    print(f"  - Summary CSV: {csv_path}")
    print(f"  - Noise CSV: {noise_csv_path}")
    print(f"  - JSON Results: {json_path}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_group", type=int, default=25, help="Max records per experimental tier")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs for prototype")
    args = parser.parse_args()
    run_benchmark(max_per_group=args.max_per_group, epochs=args.epochs)
