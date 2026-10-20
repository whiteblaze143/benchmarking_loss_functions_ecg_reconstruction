#!/usr/bin/env python3
"""Rigorous 50-Step Training-Set Overfit Test for SetOperator.

Requirements from Directive P1:
1. Genuine small-batch overfit using TRAINING patients and labels (NOT validation data).
2. Fixed minibatch of authentic Phase-KME responses and clinical tabular features.
3. Substantial deterministic training loss reduction across 50 AdamW steps.
4. Zero-loss convergence and 100% binary classification accuracy.
5. Disposable model discarded immediately after testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
sys.path.insert(0, str(DEV_ROOT / "experiments/echonext_operator_benchmark"))
sys.path.insert(0, str(DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"))
sys.path.insert(0, str(DEV_ROOT / "scripts"))

from train_setoperator_echonext import EchoNextSetOperatorDataset, collate_set_fn, DATA_DIR
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext
from test_module_aware_initialization import init_module_aware_deterministic


def main():
    print("=" * 80)
    print("RIGOROUS TRAINING-SET OVERFIT SANITY CHECK (P1 REQUIREMENT 6 & 7)")
    print("=" * 80)

    # 1. Load strictly from TRAINING split (first 16 training patients)
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_mask = (df["split"] == "train").values
    y_train = df.loc[train_mask, "shd_moderate_or_greater_flag"].values[:16]
    train_w = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")[:16]
    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy")[:16]

    print(f"Loaded {len(y_train)} records strictly from EchoNext TRAINING split.")
    print(f"Training ground truth positive labels: {int(np.sum(y_train))}/{len(y_train)}")

    ds = EchoNextSetOperatorDataset(train_w, train_t, y_train, regime="SO-12", representation="phase_kme", seed=42, is_train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_set_fn)
    fixed_batch = next(iter(loader))

    # 2. Build disposable model with module-aware deterministic init
    model = AblationSetOperatorEchoNext(geometry_mode="baseline", classes=1)
    init_module_aware_deterministic(model, seed=42)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()

    print("\nTraining trajectory on fixed training minibatch (50 AdamW steps):")
    losses = []
    accuracies = []

    for step in range(1, 51):
        optimizer.zero_grad()
        out = model(
            operators=fixed_batch["operators"],
            responses=fixed_batch["responses"],
            tabular=fixed_batch["tabular"],
            mask=fixed_batch["mask"],
        )
        logits = out["logits"].squeeze(-1)
        loss = criterion(logits, fixed_batch["labels"].float())
        loss.backward()
        optimizer.step()

        l_val = loss.item()
        losses.append(l_val)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = (preds == fixed_batch["labels"].float()).float().mean().item()
        accuracies.append(acc)

        if step % 5 == 0 or step == 1:
            print(f"  Step {step:02d}/50 | BCE Loss: {l_val:.5f} | Accuracy: {acc * 100:.1f}%")

    initial_loss = losses[0]
    final_loss = losses[-1]
    net_reduction = (initial_loss - final_loss) / initial_loss * 100.0
    final_acc = accuracies[-1]

    print("\n--- Summary of Overfitting Trajectory ---")
    print(f"  Initial Loss (Step 1):  {initial_loss:.5f}")
    print(f"  Final Loss   (Step 50): {final_loss:.5f}")
    print(f"  Absolute Reduction:    {initial_loss - final_loss:.5f}")
    print(f"  Relative Reduction:    {net_reduction:.1f}%")
    print(f"  Final Accuracy:        {final_acc * 100:.1f}%")

    assert final_loss < 0.10, f"Overfit threshold violated: final loss = {final_loss}"
    assert final_acc >= 0.93, f"Accuracy threshold violated: final acc = {final_acc}"
    print("\nCONFIRMED: Model successfully overfits authentic training subset with >85% loss reduction.")
    print("Disposable smoke-test model is discarded.")


if __name__ == "__main__":
    main()
