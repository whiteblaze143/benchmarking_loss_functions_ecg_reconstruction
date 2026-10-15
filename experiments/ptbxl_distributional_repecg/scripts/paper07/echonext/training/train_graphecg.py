"""Train and evaluate GraphECG on the official EchoNext benchmark for Binary Structural Heart Disease (SHD).

Published Protocol Reproduction:
- Architecture: GraphECG (node_dim=128, edge_dim=192, hidden_dim=192, num_layers=3, tabular_dim=7, num_classes=1)
- Spherical Harmonic Node Encoding (degree 4) + Conv1D Waveform Edge Encoding
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-6)
- Gradient clipping: max_norm=1.0
- Loss: BCEWithLogitsLoss
- Batch size: 32
- Max Epochs: 50 with Early Stopping on Validation AUROC (patience=10)
- Multi-seed support (default seeds: 42, 43, 44, 45, 46)
- Full benchmark evaluation across R01-R07 (including GECG-S subgraph and GECG-Z zero-fill)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# Set paths
REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
GRAPH_SRC = REPO_ROOT / "experiments/ptbxl_distributional_repecg/src/graphECG_author_code"
if str(GRAPH_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPH_SRC))

from graph import ECGGraphBuilder, LEAD_ORDER, LEAD_DEFINITIONS, ELECTRODE_POSITIONS, ALL_ELECTRODES
from model import GraphECG

# Disable cuDNN for Conv1d to prevent ptrDesc->finalize() descriptor bug in PyTorch/CUDA 12
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graphecg_echonext")

ECHONEXT_DIR = Path(os.environ.get("ECHONEXT_DIR", "/data/mithunmanivannan/echonext"))
OUTPUT_BASE = REPO_ROOT / "results/paper07_echonext/graphecg"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphECGDataset(Dataset):
    """EchoNext GraphECG Dataset."""

    def __init__(
        self,
        waveforms_mmap: np.ndarray,
        tabular_mmap: np.ndarray,
        labels: np.ndarray,
        builder: ECGGraphBuilder,
        lead_indices: Optional[List[int]] = None,
        zero_fill: bool = False,
        custom_edge_fn: Optional[Any] = None,
    ):
        self.waveforms = waveforms_mmap
        self.tabular = tabular_mmap
        self.labels = labels
        self.builder = builder
        self.lead_indices = lead_indices
        self.zero_fill = zero_fill
        self.custom_edge_fn = custom_edge_fn

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        # Waveform shape: (1, 2500, 12) -> transpose to (12, 2500)
        raw_wf = np.array(self.waveforms[idx, 0], dtype=np.float32).T  # (12, 2500)
        tab = np.array(self.tabular[idx], dtype=np.float32)
        label = float(self.labels[idx])

        if self.custom_edge_fn is not None:
            edges = self.custom_edge_fn(raw_wf)
            graph = self.builder.build_custom(edges, bidirectional=True)
        elif self.zero_fill and self.lead_indices is not None:
            # GECG-Z: Keep full 12 edges, but zero out unavailable leads
            wf_masked = raw_wf.copy()
            kept = set(self.lead_indices)
            for i in range(12):
                if i not in kept:
                    wf_masked[i, :] = 0.0
            graph = self.builder.build_from_array(wf_masked, lead_indices=None, bidirectional=True)
        else:
            # GECG-S: Construct subgraph using only specified leads
            graph = self.builder.build_from_array(raw_wf, lead_indices=self.lead_indices, bidirectional=True)

        return graph, torch.from_numpy(tab), torch.tensor(label, dtype=torch.float32)


def collate_graphs(batch):
    graphs = [item[0] for item in batch]
    tabs = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    batched_graphs = Batch.from_data_list(graphs)
    return batched_graphs, tabs, labels


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_graphs, tabs, labels in loader:
            batch_graphs = batch_graphs.to(device)
            tabs = tabs.to(device)
            out = model(batch_graphs, tabular=tabs)
            logits = out["logits"].squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    auroc = float(roc_auc_score(y_true, y_pred))
    auprc = float(average_precision_score(y_true, y_pred))
    return auroc, auprc, y_pred, y_true


def train_single_seed(
    seed: int,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(device_str)
    run_dir = OUTPUT_BASE / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("Starting GraphECG Training | Seed %d | Device: %s", seed, device)
    logger.info("==================================================")

    # Load metadata
    df = pd.read_csv(ECHONEXT_DIR / "echonext_metadata_100k.csv")
    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    train_labels = df[train_mask]["shd_moderate_or_greater_flag"].values.astype(int)
    val_labels = df[val_mask]["shd_moderate_or_greater_flag"].values.astype(int)
    test_labels = df[test_mask]["shd_moderate_or_greater_flag"].values.astype(int)
    test_ecg_keys = df[test_mask]["ecg_key"].values
    test_patient_keys = df[test_mask]["patient_key"].values

    logger.info("Data splits: Train=%d, Val=%d, Test=%d", len(train_labels), len(val_labels), len(test_labels))

    # Open mmap arrays
    train_w = np.load(ECHONEXT_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    train_t = np.load(ECHONEXT_DIR / "EchoNext_train_tabular_features.npy", mmap_mode="r")
    val_w = np.load(ECHONEXT_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")
    val_t = np.load(ECHONEXT_DIR / "EchoNext_val_tabular_features.npy", mmap_mode="r")
    test_w = np.load(ECHONEXT_DIR / "EchoNext_test_waveforms.npy", mmap_mode="r")
    test_t = np.load(ECHONEXT_DIR / "EchoNext_test_tabular_features.npy", mmap_mode="r")

    builder = ECGGraphBuilder()

    # Datasets and Loaders
    train_dataset = GraphECGDataset(train_w, train_t, train_labels, builder)
    val_dataset = GraphECGDataset(val_w, val_t, val_labels, builder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize GraphECG Model
    model = GraphECG(
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_layers=3,
        tabular_dim=7,
        num_classes=1,
        dropout=0.5,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_auroc = -1.0
    best_epoch = -1
    best_weights = None
    epochs_no_improve = 0
    history = []

    logger.info("Beginning training for max %d epochs (patience %d)...", epochs, patience)
    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        t0_ep = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_graphs, tabs, labels in train_loader:
            batch_graphs = batch_graphs.to(device)
            tabs = tabs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(batch_graphs, tabular=tabs)
            logits = out["logits"].squeeze(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        # Validation
        val_auroc, val_auprc, _, _ = evaluate_model(model, val_loader, device)
        ep_time = time.time() - t0_ep

        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": best_weights,
                    "val_auroc": val_auroc,
                    "val_auprc": val_auprc,
                },
                run_dir / "graphecg_best.pt",
            )
            epochs_no_improve = 0
            best_mark = " [*BEST*]"
        else:
            epochs_no_improve += 1
            best_mark = ""

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val AUROC: %.4f | Val AUPRC: %.4f | LR: %.2e | Time: %.1fs%s",
            epoch, epochs, train_loss, val_auroc, val_auprc, scheduler.get_last_lr()[0], ep_time, best_mark
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": ep_time,
        })

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d (patience %d reached).", epoch, patience)
            break

    total_train_time = time.time() - t0_train
    logger.info("Training complete in %.2f min. Best Val AUROC: %.4f at Epoch %d", total_train_time / 60, best_val_auroc, best_epoch)

    # Load best checkpoint for full test configuration benchmark
    model.load_state_dict(best_weights)
    model.eval()

    logger.info("==================================================")
    logger.info("Evaluating Best Model on Official Test Split across Configurations")
    logger.info("==================================================")

    # Lead configurations
    # 0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
    configs = {
        "R01_full12": {"leads": list(range(12)), "zero_fill": False},
        "R02_six_limb": {"leads": [0, 1, 2, 3, 4, 5], "zero_fill": False},
        "R03_leads_I_II_III": {"leads": [0, 1, 2], "zero_fill": False},
        "R04_leads_I_II": {"leads": [0, 1], "zero_fill": False},
        "R05_single_I": {"leads": [0], "zero_fill": False},
        "R06_single_II": {"leads": [1], "zero_fill": False},
        # Zero-filled comparators (GECG-Z)
        "R02_six_limb_zerofill": {"leads": [0, 1, 2, 3, 4, 5], "zero_fill": True},
        "R03_leads_I_II_III_zerofill": {"leads": [0, 1, 2], "zero_fill": True},
        "R04_leads_I_II_zerofill": {"leads": [0, 1], "zero_fill": True},
        "R05_single_I_zerofill": {"leads": [0], "zero_fill": True},
        "R06_single_II_zerofill": {"leads": [1], "zero_fill": True},
    }

    test_results = {}
    preds_dict = {
        "ecg_key": test_ecg_keys,
        "patient_key": test_patient_keys,
        "ground_truth_shd": test_labels,
    }

    for cfg_name, cfg in configs.items():
        ds = GraphECGDataset(test_w, test_t, test_labels, builder, lead_indices=cfg["leads"], zero_fill=cfg["zero_fill"])
        loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_graphs, num_workers=4)
        auroc, auprc, probs, _ = evaluate_model(model, loader, device)
        logger.info("Test Config %-28s | AUROC: %.4f | AUPRC: %.4f", cfg_name, auroc, auprc)
        test_results[cfg_name] = {"auroc": round(auroc, 5), "auprc": round(auprc, 5)}
        preds_dict[f"prob_{cfg_name}"] = probs

    # R07: Synthetic V3-V2 differential
    v2_idx = 7  # V2
    v3_idx = 8  # V3
    def v3_minus_v2_fn(ecg_12):
        sig = ecg_12[v3_idx] - ecg_12[v2_idx]
        return [("V2", "V3", sig)]

    r07_ds = GraphECGDataset(test_w, test_t, test_labels, builder, custom_edge_fn=v3_minus_v2_fn)
    r07_loader = DataLoader(r07_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_graphs, num_workers=4)
    r07_auroc, r07_auprc, r07_probs, _ = evaluate_model(model, r07_loader, device)
    logger.info("Test Config %-28s | AUROC: %.4f | AUPRC: %.4f", "R07_v3minus_v2", r07_auroc, r07_auprc)
    test_results["R07_v3minus_v2"] = {"auroc": round(r07_auroc, 5), "auprc": round(r07_auprc, 5)}
    preds_dict["prob_R07_v3minus_v2"] = r07_probs

    # Save outputs
    pd.DataFrame(preds_dict).to_parquet(run_dir / "test_predictions.parquet", index=False)
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "total_train_time_min": total_train_time / 60,
        "test_benchmark": test_results,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved seed summary to %s", run_dir / "summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_single_seed(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
