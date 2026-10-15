"""Evaluate official published EchoNext-Mini baseline on EchoNext test split.

Protocol:
- Model: ResNet1dWithTabular (filter_size=16, len_tabular=7, num_classes=12)
- Pretrained weights: IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt
- Test Split: N=5,442 official EchoNext test records
- Target: shd_moderate_or_greater_flag (column index 11 of 12 classes)
- Evaluates configurations R01-R07 (full 12 leads, 6 limb, I-II-III, I-II, single I, single II, V3-V2 differential)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

# Disable cuDNN for Conv1d to prevent ptrDesc->finalize() descriptor bug in PyTorch/CUDA 12
torch.backends.cudnn.enabled = False

# Add IntroECG to path
INTROECG_DIR = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/IntroECG/7-EchoNext Minimodel")
if str(INTROECG_DIR) not in sys.path:
    sys.path.insert(0, str(INTROECG_DIR))

from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("echonext_mini_eval")

ECHONEXT_DIR = Path("/home/mithunmanivannan/data/echonext")
WEIGHTS_PATH = INTROECG_DIR / "models/echonext_multilabel_minimodel/weights.pt"
OUTPUT_DIR = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/results/paper07_echonext/echonext_mini")

# Standard 12-lead indices:
# 0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
CONFIGS = {
    "R01_full12": {"leads": list(range(12)), "type": "subset"},
    "R02_six_limb": {"leads": [0, 1, 2, 3, 4, 5], "type": "subset"},
    "R03_leads_I_II_III": {"leads": [0, 1, 2], "type": "subset"},
    "R04_leads_I_II": {"leads": [0, 1], "type": "subset"},
    "R05_single_I": {"leads": [0], "type": "subset"},
    "R06_single_II": {"leads": [1], "type": "subset"},
    "R07_v3minus_v2_at_v3": {"type": "differential", "target_channel": 8},  # V3 channel
    "R07_v3minus_v2_at_lead1": {"type": "differential", "target_channel": 0},  # Lead I channel
    "R07_v3minus_v2_at_lead2": {"type": "differential", "target_channel": 1},  # Lead II channel
}


def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n_bootstraps: int = 2000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)
    bootstrapped_scores = np.array(bootstrapped_scores)
    return {
        "auroc_mean": float(np.mean(bootstrapped_scores)),
        "auroc_std": float(np.std(bootstrapped_scores)),
        "ci_lower": float(np.percentile(bootstrapped_scores, 2.5)),
        "ci_upper": float(np.percentile(bootstrapped_scores, 97.5)),
    }


def evaluate_configuration(
    model: torch.nn.Module,
    test_w_mmap: np.ndarray,
    test_t_mmap: np.ndarray,
    cfg: Dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n_samples = len(test_w_mmap)
    cfg_type = cfg.get("type", "subset")
    all_probs = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            # Load contiguous block directly as float32
            wf = np.array(test_w_mmap[start:end], dtype=np.float32)
            tab = np.array(test_t_mmap[start:end], dtype=np.float32)

            if cfg_type == "subset":
                kept_leads = set(cfg["leads"])
                for lead_idx in range(12):
                    if lead_idx not in kept_leads:
                        wf[:, 0, :, lead_idx] = 0.0
            elif cfg_type == "differential":
                # V3 is index 8, V2 is index 7
                v3_minus_v2 = wf[:, 0, :, 8] - wf[:, 0, :, 7]
                wf[:, 0, :, :] = 0.0  # zero all
                target_chan = cfg["target_channel"]
                wf[:, 0, :, target_chan] = v3_minus_v2

            x_tensor = torch.from_numpy(wf).to(device)
            tab_tensor = torch.from_numpy(tab).to(device)
            logits = model((x_tensor, tab_tensor))
            # Column 11 is composite SHD
            shd_probs = torch.sigmoid(logits[:, 11]).cpu().numpy()
            all_probs.append(shd_probs)

    return np.concatenate(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    logger.info("Loading EchoNext test metadata...")
    df_all = pd.read_csv(ECHONEXT_DIR / "echonext_metadata_100k.csv")
    df_test = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
    y_test = df_test["shd_moderate_or_greater_flag"].values.astype(int)
    ecg_keys = df_test["ecg_key"].values
    patient_keys = df_test["patient_key"].values

    logger.info("Test records: %d, Positive SHD: %d (%.2f%%)", len(y_test), y_test.sum(), 100 * y_test.mean())

    test_w_path = ECHONEXT_DIR / "EchoNext_test_waveforms.npy"
    test_t_path = ECHONEXT_DIR / "EchoNext_test_tabular_features.npy"

    logger.info("Loading test waveforms and tabular features from local disk into RAM...")
    t0_load = time.time()
    test_w_ram = np.load(test_w_path).astype(np.float32)
    test_t_ram = np.load(test_t_path).astype(np.float32)
    logger.info("Test set fully loaded in RAM in %.2fs. Shapes: wf=%s, tab=%s", time.time() - t0_load, test_w_ram.shape, test_t_ram.shape)

    logger.info("Building ResNet1dWithTabular model...")
    model = ResNet1dWithTabular(
        len_tabular_feature_vector=7,
        filter_size=16,
        num_classes=12,
    )

    logger.info("Loading official checkpoint from %s...", WEIGHTS_PATH)
    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)

    model.to(device)
    model.eval()
    logger.info("Model loaded and moved to %s.", device)

    results = {}
    predictions_dict = {
        "ecg_key": ecg_keys,
        "patient_key": patient_keys,
        "ground_truth_shd": y_test,
    }

    for cfg_name, cfg in CONFIGS.items():
        logger.info("Evaluating configuration: %s...", cfg_name)
        t0 = time.time()
        y_prob = evaluate_configuration(model, test_w_ram, test_t_ram, cfg, batch_size=args.batch_size, device=device)
        eval_time = time.time() - t0

        auroc = float(roc_auc_score(y_test, y_prob))
        auprc = float(average_precision_score(y_test, y_prob))

        logger.info(
            "Config %-25s | AUROC: %.4f | AUPRC: %.4f | Time: %.2fs",
            cfg_name, auroc, auprc, eval_time
        )

        ci_info = bootstrap_auroc(y_test, y_prob, n_bootstraps=args.n_bootstraps)
        results[cfg_name] = {
            "auroc": round(auroc, 5),
            "auprc": round(auprc, 5),
            "ci_95": [round(ci_info["ci_lower"], 5), round(ci_info["ci_upper"], 5)],
            "auroc_std": round(ci_info["auroc_std"], 5),
            "inference_time_sec": round(eval_time, 2),
        }
        predictions_dict[f"prob_{cfg_name}"] = y_prob

    # Save predictions to parquet
    preds_df = pd.DataFrame(predictions_dict)
    preds_path = OUTPUT_DIR / "echonext_mini_test_predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)
    logger.info("Saved aligned predictions to %s", preds_path)

    summary_file = OUTPUT_DIR / "echonext_mini_results.json"
    summary_file.write_text(json.dumps(results, indent=2))
    logger.info("Saved results summary to %s", summary_file)

    print("\n" + "=" * 70)
    print("ECHONEXT-MINI OFFICIAL BASELINE REPLICATION RESULTS (TEST SET, N=5,442)")
    print("=" * 70)
    print(f"{'Configuration':<28} | {'AUROC':<8} | {'95% CI':<18} | {'AUPRC':<8}")
    print("-" * 70)
    for k, v in results.items():
        ci_str = f"[{v['ci_95'][0]:.4f}, {v['ci_95'][1]:.4f}]"
        print(f"{k:<28} | {v['auroc']:<8.4f} | {ci_str:<18} | {v['auprc']:<8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
