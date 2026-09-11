"""Linear Probe Evaluation Module for GRAIL-ECG.

Fits and evaluates linear classifiers on frozen representations Z in R^{96}
to test linear accessibility and clinical sufficiency without fine-tuning the encoder.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    """Linear classifier on frozen representations."""

    def __init__(self, in_features: int = 96, num_classes: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_spec: float = 0.95,
    target_sens: float = 0.95,
) -> dict[str, float]:
    """Computes AUROC, AUPRC, Sens@95Sp, Spec@95Se, Brier score, and ECE."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "sens_at_95spec": float("nan"),
            "spec_at_95sens": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Threshold sweeps for operating points
    thresholds = np.sort(y_prob)
    sens_list = []
    spec_list = []
    for thresh in thresholds:
        pred = (y_prob >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    sens_arr = np.array(sens_list)
    spec_arr = np.array(spec_list)

    # Sens @ 95% Spec
    valid_spec = np.where(spec_arr >= target_spec)[0]
    sens_at_95spec = float(np.max(sens_arr[valid_spec])) if len(valid_spec) > 0 else 0.0

    # Spec @ 95% Sens
    valid_sens = np.where(sens_arr >= target_sens)[0]
    spec_at_95sens = float(np.max(spec_arr[valid_sens])) if len(valid_sens) > 0 else 0.0

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sens_at_95spec": sens_at_95spec,
        "spec_at_95sens": spec_at_95sens,
        "brier": brier,
        "ece": float(ece),
    }


def fit_and_evaluate_linear_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, float]:
    """Fits a logistic linear probe on frozen Z and evaluates on validation."""
    D = z_train.shape[1]
    num_classes = y_train.shape[1] if y_train.ndim > 1 else 1

    model = LinearProbe(in_features=D, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(z_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(num_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(z_val.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    y_val_np = y_val.cpu().numpy()
    if num_classes == 1:
        return compute_binary_metrics(y_val_np, val_probs)
    else:
        # Macro average across classes
        class_metrics = [
            compute_binary_metrics(y_val_np[:, c], val_probs[:, c])
            for c in range(num_classes)
        ]
        macro_auroc = float(np.nanmean([m["auroc"] for m in class_metrics]))
        macro_auprc = float(np.nanmean([m["auprc"] for m in class_metrics]))
        macro_brier = float(np.nanmean([m["brier"] for m in class_metrics]))
        return {
            "macro_auroc": macro_auroc,
            "macro_auprc": macro_auprc,
            "macro_brier": macro_brier,
        }
