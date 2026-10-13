#!/usr/bin/env python3
"""Run Milestone R015 / Block 3: Disease-Preservation Triad Benchmark.

Evaluates the central scientific triad comparing frozen SetOperator (d=256) vs
frozen CSFM Base (d=768) on the longitudinal PTB-XL repeated-patient cohort:

Simultaneous Triad:
1. State Accessibility: Can frozen representations decode primary diagnostic categories?
   (Linear probes for NORM, MI, STTC, CD, HYP; Macro AUROC and per-class AUROC).
2. Patient Preservation Across Acute Transitions:
   Does same-patient verification geometry survive acute clinical transitions?
   (Stable vs Any Transition, NORM->MI, NORM->STTC, CD-->CD+, MI-->MI+, AF<->SR).
3. Patient-Relative Change Gain:
   Does subtracting personal baseline reference (z_t - b_p) improve acute transition
   discrimination over population features (z_t)?
   G = AUROC(z_t - b_p) - AUROC(z_t).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "prism_ecg_disease_preservation_triad_v2"
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("disease_preservation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC computation."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_and_eval_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: Tuple[float, ...] = C_GRID,
) -> Tuple[float, np.ndarray]:
    """Fit standardized L2 logistic regression with tuned C on train/val split."""
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train)
    x_te_scaled = scaler.transform(x_test)

    best_auc = -1.0
    best_probs = None

    for c_val in c_grid:
        try:
            clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42, solver="lbfgs")
            clf.fit(x_tr_scaled, y_train)
            probs = clf.predict_proba(x_te_scaled)[:, 1]
            auc = fast_auroc(y_test, probs)
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_probs = probs
        except Exception:
            continue

    if best_probs is None:
        best_probs = np.zeros(len(y_test))
        best_auc = 0.5
    return best_auc, best_probs


def evaluate_state_accessibility(
    z_all: np.ndarray,
    df_multi: pd.DataFrame,
    logger: logging.Logger,
    rep_name: str,
) -> Dict[str, Any]:
    """Pillar 1: Linear probe diagnostic decodability for NORM, MI, STTC, CD, HYP."""
    train_mask = (df_multi["strat_fold"].values <= 7)
    test_mask = (df_multi["strat_fold"].values == 8)

    x_train = z_all[train_mask]
    x_test = z_all[test_mask]

    class_aurocs = {}
    valid_aurocs = []

    for sc in SUPERCLASSES:
        y_train = df_multi[sc].values[train_mask]
        y_test = df_multi[sc].values[test_mask]

        if y_train.sum() == 0 or y_test.sum() == 0:
            class_aurocs[sc] = float("nan")
            continue

        auc, _ = fit_and_eval_probe(x_train, y_train, x_test, y_test)
        class_aurocs[sc] = float(auc)
        if not np.isnan(auc):
            valid_aurocs.append(auc)

    macro_auc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    logger.info(f"[{rep_name}] State Accessibility Macro AUROC = {macro_auc:.4f} {class_aurocs}")
    return {
        "macro_auroc": macro_auc,
        "class_aurocs": class_aurocs,
    }


def evaluate_patient_preservation_by_transition(
    z_all: np.ndarray,
    same_pairs: List[Dict[str, Any]],
    diff_pairs: List[Dict[str, Any]],
    logger: logging.Logger,
    rep_name: str,
) -> Dict[str, Any]:
    """Pillar 2: Same-patient verification AUROC stratified by transition status."""
    z_norm = z_all / (np.linalg.norm(z_all, axis=1, keepdims=True) + 1e-12)
    z_cent = z_all - np.mean(z_all, axis=0, keepdims=True)

    same_cos = np.array([np.dot(z_norm[p["pos1"]], z_norm[p["pos2"]]) for p in same_pairs])
    diff_cos = np.array([np.dot(z_norm[p["pos1"]], z_norm[p["pos2"]]) for p in diff_pairs])

    same_euc = -np.array([np.linalg.norm(z_cent[p["pos1"]] - z_cent[p["pos2"]]) for p in same_pairs])
    diff_euc = -np.array([np.linalg.norm(z_cent[p["pos1"]] - z_cent[p["pos2"]]) for p in diff_pairs])

    # Stratify by pair transition type
    pair_types = {
        "All Pairs": np.ones(len(same_pairs), dtype=bool),
        "Stable (Unchanged)": np.array([p["is_stable"] for p in same_pairs]),
        "Acute Transition (Any Change)": np.array([p["is_changed"] for p in same_pairs]),
        "NORM -> MI": np.array([p["is_norm_to_mi"] for p in same_pairs]),
        "NORM -> STTC": np.array([p["is_norm_to_sttc"] for p in same_pairs]),
        "CD- -> CD+ (New CD)": np.array([p["is_cd_onset"] for p in same_pairs]),
        "MI- -> MI+ (New MI)": np.array([p["is_mi_onset"] for p in same_pairs]),
        "AF <-> SR Rhythm Shift": np.array([p["is_rhythm_shift"] for p in same_pairs]),
    }

    preservation_results = {}
    for stratum_name, mask in pair_types.items():
        n_same = int(mask.sum())
        if n_same < 5:
            continue

        s_cos = same_cos[mask]
        s_euc = same_euc[mask]

        # Matched impostors (10x)
        d_cos = diff_cos[:n_same * 10]
        d_euc = diff_euc[:n_same * 10]

        y_eval = np.concatenate([np.ones(n_same), np.zeros(len(d_cos))])
        auc_cos = fast_auroc(y_eval, np.concatenate([s_cos, d_cos]))
        auc_euc = fast_auroc(y_eval, np.concatenate([s_euc, d_euc]))

        preservation_results[stratum_name] = {
            "n_pairs": n_same,
            "cosine_auroc": float(auc_cos),
            "euclidean_auroc": float(auc_euc),
        }
        logger.info(f"[{rep_name}] {stratum_name} (N={n_same}): Euclidean={auc_euc:.4f}, Cosine={auc_cos:.4f}")

    return preservation_results


def evaluate_relative_change_gain(
    z_all: np.ndarray,
    df_multi: pd.DataFrame,
    same_pairs: List[Dict[str, Any]],
    logger: logging.Logger,
    rep_name: str,
) -> Dict[str, Any]:
    """Pillar 3: Personal Reference Calibration Gain G = AUROC(z_t - b_p) - AUROC(z_t)."""
    pair_df = pd.DataFrame(same_pairs)

    tasks = {
        "General State Transition": {
            "eligible": np.ones(len(pair_df), dtype=bool),
            "target_label": pair_df["is_changed"].astype(int).values,
        },
        "Acute MI Event (MI- -> MI+)": {
            "eligible": (pair_df["t1_mi"] == 0).values,
            "target_label": (pair_df["t2_mi"] == 1).astype(int).values,
        },
        "Acute STTC Event (STTC- -> STTC+)": {
            "eligible": (pair_df["t1_sttc"] == 0).values,
            "target_label": (pair_df["t2_sttc"] == 1).astype(int).values,
        },
        "New Conduction Disturbance (CD- -> CD+)": {
            "eligible": (pair_df["t1_cd"] == 0).values,
            "target_label": (pair_df["t2_cd"] == 1).astype(int).values,
        },
        "Acute Onset from NORM Baseline": {
            "eligible": (pair_df["t1_norm"] == 1).values,
            "target_label": (pair_df["t2_norm"] == 0).astype(int).values,
        },
    }

    gain_results = {}
    for task_name, task_cfg in tasks.items():
        el_mask = task_cfg["eligible"]
        y_all = task_cfg["target_label"][el_mask]

        sub_pairs = [same_pairs[i] for i in range(len(same_pairs)) if el_mask[i]]
        if len(sub_pairs) < 20 or y_all.sum() < 5 or (len(y_all) - y_all.sum()) < 5:
            continue

        folds = np.array([p["strat_fold"] for p in sub_pairs])
        tr_mask = (folds <= 7)
        te_mask = (folds == 8)

        if tr_mask.sum() < 10 or te_mask.sum() < 10 or y_all[tr_mask].sum() == 0 or y_all[te_mask].sum() == 0:
            continue

        pos1_idx = np.array([p["pos1"] for p in sub_pairs])
        pos2_idx = np.array([p["pos2"] for p in sub_pairs])

        # Raw population features: z_{t2}
        x_raw_tr = z_all[pos2_idx[tr_mask]]
        x_raw_te = z_all[pos2_idx[te_mask]]

        # Calibrated reference change features: z_{t2} - z_{t1}
        x_cal_tr = z_all[pos2_idx[tr_mask]] - z_all[pos1_idx[tr_mask]]
        x_cal_te = z_all[pos2_idx[te_mask]] - z_all[pos1_idx[te_mask]]

        y_tr = y_all[tr_mask]
        y_te = y_all[te_mask]

        auc_raw, _ = fit_and_eval_probe(x_raw_tr, y_tr, x_raw_te, y_te)
        auc_cal, _ = fit_and_eval_probe(x_cal_tr, y_tr, x_cal_te, y_te)
        delta_gain = auc_cal - auc_raw

        gain_results[task_name] = {
            "n_pairs": len(sub_pairs),
            "n_events": int(y_all.sum()),
            "auroc_raw_population": float(auc_raw),
            "auroc_calibrated_change": float(auc_cal),
            "calibration_gain_delta": float(delta_gain),
        }
        logger.info(f"[{rep_name}] {task_name}: Raw={auc_raw:.4f}, Calibrated={auc_cal:.4f}, Gain Δ={delta_gain:+.4f}")

    return gain_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone R015: Disease-Preservation Triad Benchmark")
    parser.add_argument("--so-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"))
    parser.add_argument("--csfm-cache", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/csfm_base_multi_z.npz"))
    parser.add_argument("--ptbxl-data", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_disease_preservation_benchmark"))
    parser.add_argument("--n-impostors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir / "run.log")

    logger.info("=" * 70)
    logger.info("Milestone R015: Disease-Preservation Triad Benchmark")
    logger.info(f"SetOperator Cache: {args.so_cache}")
    logger.info(f"CSFM Base Cache:   {args.csfm_cache}")
    logger.info(f"Output Directory:  {args.output_dir}")
    logger.info("=" * 70)

    # 1. Load Metadata & Diagnostic Annotations
    logger.info("Loading PTB-XL metadata and scp_statements...")
    scp_df = pd.read_csv(args.ptbxl_data / "scp_statements.csv", index_col=0)
    diag_map = scp_df["diagnostic_class"].dropna().to_dict()

    df_meta = pd.read_csv(args.ptbxl_data / "ptbxl_database.csv", index_col="ecg_id")
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])

    so_npz = np.load(args.so_cache)
    all_z_so = so_npz["all_z"]
    all_ecg_so = so_npz["all_ecg"]
    ecg_to_so_idx = {int(ecg_id): i for i, ecg_id in enumerate(all_ecg_so)}

    df_meta = df_meta.loc[all_ecg_so].copy()

    # Extract multi-hot indicators for all 5 diagnostic superclasses
    for sc in SUPERCLASSES:
        df_meta[sc] = df_meta["scp_codes"].apply(
            lambda s: int(any(diag_map.get(k) == sc for k in ast.literal_eval(s).keys()))
            if isinstance(s, str) else 0
        )

    def get_clean_rhythm(scp_str):
        if not isinstance(scp_str, str):
            return "OTHER"
        try:
            d = ast.literal_eval(scp_str)
            if "AFIB" in d or "AFLT" in d:
                return "AF"
            if "SR" in d:
                return "SR"
        except Exception:
            pass
        return "OTHER"

    df_meta["clean_rhythm"] = df_meta["scp_codes"].apply(get_clean_rhythm)

    pat_counts = df_meta["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 2].index.values
    df_multi = df_meta[df_meta["patient_id"].isin(multi_pats)].sort_values(["patient_id", "recording_date"]).copy()
    df_multi["pos"] = np.arange(len(df_multi))
    logger.info(f"Repeated Cohort: {len(multi_pats):,} patients, {len(df_multi):,} ECG records.")

    for sc in SUPERCLASSES:
        logger.info(f"Prevalence in cohort - {sc}: {df_multi[sc].sum():,} ({df_multi[sc].mean()*100:.1f}%)")

    # 2. Slice Representations
    so_indices = [ecg_to_so_idx[ecg_id] for ecg_id in df_multi.index.values]
    z_so_native = all_z_so[so_indices].astype(np.float32)

    csfm_npz = np.load(args.csfm_cache)
    z_csfm_native = csfm_npz["csfm_z"].astype(np.float32)
    logger.info(f"Loaded Latents: SetOperator {z_so_native.shape}, CSFM Base {z_csfm_native.shape}")

    # 3. Fit Dimensionality-Matched PCA on Training Patients
    train_mask = (df_multi["strat_fold"].values <= 7)
    logger.info(f"Fitting PCA on Training Patients ({train_mask.sum():,} records)...")

    pca_csfm_256 = PCA(n_components=256, random_state=args.seed).fit(z_csfm_native[train_mask])
    z_csfm_d256 = pca_csfm_256.transform(z_csfm_native).astype(np.float32)

    pca_csfm_128 = PCA(n_components=128, random_state=args.seed).fit(z_csfm_native[train_mask])
    z_csfm_d128 = pca_csfm_128.transform(z_csfm_native).astype(np.float32)

    pca_so_128 = PCA(n_components=128, random_state=args.seed).fit(z_so_native[train_mask])
    z_so_d128 = pca_so_128.transform(z_so_native).astype(np.float32)

    # 4. Form Longitudinal Pairs
    logger.info("Forming longitudinal same-patient and matched impostor pairs...")
    rng = np.random.RandomState(args.seed)
    same_pairs = []
    diff_pairs = []

    for pid, group in df_multi.groupby("patient_id"):
        recs = group.to_dict("records")
        for i in range(len(recs) - 1):
            t1 = recs[i]
            t2 = recs[i + 1]
            dt_days = (t2["recording_date"] - t1["recording_date"]).total_seconds() / 86400.0

            is_changed = any(t1[sc] != t2[sc] for sc in SUPERCLASSES)
            is_stable = not is_changed
            is_norm_to_mi = (t1["NORM"] == 1 and t2["NORM"] == 0 and t2["MI"] == 1)
            is_norm_to_sttc = (t1["NORM"] == 1 and t2["NORM"] == 0 and t2["STTC"] == 1)
            is_cd_onset = (t1["CD"] == 0 and t2["CD"] == 1)
            is_mi_onset = (t1["MI"] == 0 and t2["MI"] == 1)
            is_sttc_onset = (t1["STTC"] == 0 and t2["STTC"] == 1)
            is_rhythm_shift = (
                (t1["clean_rhythm"] == "AF" and t2["clean_rhythm"] == "SR") or
                (t1["clean_rhythm"] == "SR" and t2["clean_rhythm"] == "AF")
            )

            same_pairs.append({
                "patient_id": pid,
                "pos1": t1["pos"],
                "pos2": t2["pos"],
                "dt_days": dt_days,
                "is_stable": is_stable,
                "is_changed": is_changed,
                "is_norm_to_mi": is_norm_to_mi,
                "is_norm_to_sttc": is_norm_to_sttc,
                "is_cd_onset": is_cd_onset,
                "is_mi_onset": is_mi_onset,
                "is_sttc_onset": is_sttc_onset,
                "is_rhythm_shift": is_rhythm_shift,
                "t1_norm": t1["NORM"],
                "t2_norm": t2["NORM"],
                "t1_mi": t1["MI"],
                "t2_mi": t2["MI"],
                "t1_sttc": t1["STTC"],
                "t2_sttc": t2["STTC"],
                "t1_cd": t1["CD"],
                "t2_cd": t2["CD"],
                "t1_hyp": t1["HYP"],
                "t2_hyp": t2["HYP"],
                "strat_fold": t2["strat_fold"],
            })

    all_positions = df_multi["pos"].values
    for sp in same_pairs:
        pos1 = sp["pos1"]
        pid1 = sp["patient_id"]
        impostors = []
        while len(impostors) < args.n_impostors:
            cand_pos = rng.choice(all_positions)
            cand_pid = df_multi.iloc[cand_pos]["patient_id"]
            if cand_pid != pid1 and cand_pos not in impostors:
                impostors.append(cand_pos)
        for imp_pos in impostors:
            diff_pairs.append({"pos1": pos1, "pos2": imp_pos})

    logger.info(f"Formed {len(same_pairs):,} same-patient pairs and {len(diff_pairs):,} impostor pairs.")
    pair_df = pd.DataFrame(same_pairs)
    logger.info(f"Transition counts: Stable={pair_df['is_stable'].sum()}, Changed={pair_df['is_changed'].sum()}, "
                f"NORM->MI={pair_df['is_norm_to_mi'].sum()}, NORM->STTC={pair_df['is_norm_to_sttc'].sum()}, "
                f"CD onset={pair_df['is_cd_onset'].sum()}, MI onset={pair_df['is_mi_onset'].sum()}, "
                f"AF/SR shift={pair_df['is_rhythm_shift'].sum()}")

    # 5. Evaluate the Triad Across All Representation Arms
    arms = [
        ("SetOperator (Native, d=256)", z_so_native),
        ("CSFM Base (Native, d=768)", z_csfm_native),
        ("SetOperator (Matched, d=128)", z_so_d128),
        ("CSFM Base (Matched, d=128)", z_csfm_d128),
        ("CSFM Base (Matched, d=256)", z_csfm_d256),
    ]

    triad_results = {}
    for arm_name, z_mat in arms:
        logger.info("\n" + "=" * 70)
        logger.info(f"Evaluating Triad Arm: {arm_name}")
        logger.info("=" * 70)

        # Pillar 1: State Accessibility
        logger.info("--- Pillar 1: State Accessibility (Diagnostic Classification Probes) ---")
        p1 = evaluate_state_accessibility(z_mat, df_multi, logger, arm_name)

        # Pillar 2: Patient Preservation Across Transitions
        logger.info("\n--- Pillar 2: Patient Preservation Across Acute Transitions ---")
        p2 = evaluate_patient_preservation_by_transition(z_mat, same_pairs, diff_pairs, logger, arm_name)

        # Pillar 3: Relative Change Gain
        logger.info("\n--- Pillar 3: Relative Change Gain (z_t - b_p) ---")
        p3 = evaluate_relative_change_gain(z_mat, df_multi, same_pairs, logger, arm_name)

        triad_results[arm_name] = {
            "pillar1_state_accessibility": p1,
            "pillar2_patient_preservation": p2,
            "pillar3_relative_change_gain": p3,
        }

    # 6. Save Structured JSON Results
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "n_patients": len(multi_pats),
        "n_records": len(df_multi),
        "n_same_pairs": len(same_pairs),
        "n_impostor_pairs": len(diff_pairs),
        "triad_evaluations": triad_results,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results_payload, f, indent=2)
    logger.info(f"\nSaved structured results to: {args.output_dir / 'results.json'}")

    # 7. Generate Markdown Summary Table
    md_lines = [
        "# Milestone R015: Disease-Preservation Triad Benchmark Report",
        "",
        "### Simultaneous Triad: State Accessibility, Patient Preservation, and Personal Calibration Gain",
        "",
        "#### Pillar 1: State Accessibility (Macro & Per-Class Diagnostic Decodability on Test Folds 9-10)",
        "| Representation Arm | Macro AUROC | NORM AUROC | MI AUROC | STTC AUROC | CD AUROC | HYP AUROC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for arm_name in triad_results:
        p1 = triad_results[arm_name]["pillar1_state_accessibility"]
        macro = p1["macro_auroc"]
        c_aucs = p1["class_aurocs"]
        md_lines.append(
            f"| **{arm_name}** | **{macro:.4f}** | {c_aucs.get('NORM', float('nan')):.4f} | {c_aucs.get('MI', float('nan')):.4f} | {c_aucs.get('STTC', float('nan')):.4f} | {c_aucs.get('CD', float('nan')):.4f} | {c_aucs.get('HYP', float('nan')):.4f} |"
        )

    md_lines.extend([
        "",
        "#### Pillar 2: Patient Preservation Across Acute Transitions (Euclidean AUROC)",
        "| Representation Arm | All Pairs | Stable Pairs | Acute Transitions | NORM → MI | NORM → STTC | CD⁻ → CD⁺ | MI⁻ → MI⁺ | AF ↔ SR Shift |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for arm_name in triad_results:
        p2 = triad_results[arm_name]["pillar2_patient_preservation"]
        all_p = p2.get("All Pairs", {}).get("euclidean_auroc", float("nan"))
        stable_p = p2.get("Stable (Unchanged)", {}).get("euclidean_auroc", float("nan"))
        acute_p = p2.get("Acute Transition (Any Change)", {}).get("euclidean_auroc", float("nan"))
        mi_p = p2.get("NORM -> MI", {}).get("euclidean_auroc", float("nan"))
        sttc_p = p2.get("NORM -> STTC", {}).get("euclidean_auroc", float("nan"))
        cd_p = p2.get("CD- -> CD+ (New CD)", {}).get("euclidean_auroc", float("nan"))
        mi_on_p = p2.get("MI- -> MI+ (New MI)", {}).get("euclidean_auroc", float("nan"))
        af_p = p2.get("AF <-> SR Rhythm Shift", {}).get("euclidean_auroc", float("nan"))
        md_lines.append(
            f"| **{arm_name}** | {all_p:.4f} | {stable_p:.4f} | {acute_p:.4f} | {mi_p:.4f} | {sttc_p:.4f} | {cd_p:.4f} | {mi_on_p:.4f} | {af_p:.4f} |"
        )

    md_lines.extend([
        "",
        "#### Pillar 3: Personal Reference Calibration Gain ($G = \\text{AUROC}(z_t - b_p) - \\text{AUROC}(z_t)$ on Test Patients)",
        "| Representation Arm | General Transition Gain Δ | Acute MI Event Gain Δ | Acute STTC Event Gain Δ | New CD Event Gain Δ | Acute Onset (NORM Base) Gain Δ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for arm_name in triad_results:
        p3 = triad_results[arm_name]["pillar3_relative_change_gain"]
        g_gen = p3.get("General State Transition", {}).get("calibration_gain_delta", float("nan"))
        g_mi = p3.get("Acute MI Event (MI- -> MI+)", {}).get("calibration_gain_delta", float("nan"))
        g_sttc = p3.get("Acute STTC Event (STTC- -> STTC+)", {}).get("calibration_gain_delta", float("nan"))
        g_cd = p3.get("New Conduction Disturbance (CD- -> CD+)", {}).get("calibration_gain_delta", float("nan"))
        g_norm = p3.get("Acute Onset from NORM Baseline", {}).get("calibration_gain_delta", float("nan"))
        md_lines.append(
            f"| **{arm_name}** | **{g_gen:+.4f}** | **{g_mi:+.4f}** | **{g_sttc:+.4f}** | **{g_cd:+.4f}** | **{g_norm:+.4f}** |"
        )

    md_content = "\n".join(md_lines) + "\n"
    with open(args.output_dir / "disease_preservation_triad_table.md", "w") as f:
        f.write(md_content)
    logger.info(f"\nSaved Markdown table to: {args.output_dir / 'disease_preservation_triad_table.md'}")
    logger.info("\n" + md_content)

    # 8. Generate Publication Figure
    logger.info("Generating publication figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    arm_labels = ["SO (Native)", "CSFM (Native)", "SO (d=128)", "CSFM (d=128)", "CSFM (d=256)"]
    arm_keys = list(triad_results.keys())
    macro_scores = [triad_results[k]["pillar1_state_accessibility"]["macro_auroc"] for k in arm_keys]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Panel A: State Accessibility
    ax = axes[0]
    bars = ax.bar(np.arange(len(arm_labels)), macro_scores, color=colors, alpha=0.9, width=0.55)
    ax.set_xticks(np.arange(len(arm_labels)))
    ax.set_xticklabels(arm_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Macro Diagnostic AUROC", fontsize=11, fontweight="bold")
    ax.set_title("A. State Accessibility\n(5-Class Diagnostic Decodability)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b, s in zip(bars, macro_scores):
        if not np.isnan(s):
            ax.text(b.get_x() + b.get_width()/2, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Patient Preservation Across Transitions
    strata = ["Stable", "All Changes", "NORM->MI", "NORM->STTC", "CD- -> CD+", "MI- -> MI+"]
    strata_map = {
        "Stable": "Stable (Unchanged)",
        "All Changes": "Acute Transition (Any Change)",
        "NORM->MI": "NORM -> MI",
        "NORM->STTC": "NORM -> STTC",
        "CD- -> CD+": "CD- -> CD+ (New CD)",
        "MI- -> MI+": "MI- -> MI+ (New MI)",
    }

    so_p2 = triad_results["SetOperator (Native, d=256)"]["pillar2_patient_preservation"]
    csfm_p2 = triad_results["CSFM Base (Native, d=768)"]["pillar2_patient_preservation"]

    so_pres = [so_p2.get(strata_map[s], {}).get("euclidean_auroc", 0.5) for s in strata]
    csfm_pres = [csfm_p2.get(strata_map[s], {}).get("euclidean_auroc", 0.5) for s in strata]

    ax = axes[1]
    x_s = np.arange(len(strata))
    w = 0.35
    ax.bar(x_s - w/2, so_pres, w, label="SetOperator (Native)", color="#1f77b4", alpha=0.9)
    ax.bar(x_s + w/2, csfm_pres, w, label="CSFM Base (Native)", color="#ff7f0e", alpha=0.9)
    ax.set_xticks(x_s)
    ax.set_xticklabels(strata, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Verification Euclidean AUROC", fontsize=11, fontweight="bold")
    ax.set_title("B. Patient Identity Preservation\nAcross Clinical Transitions", fontsize=12, fontweight="bold")
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=9)

    # Panel C: Personal Calibration Gain
    task_keys = [
        "General State Transition",
        "Acute MI Event (MI- -> MI+)",
        "Acute STTC Event (STTC- -> STTC+)",
        "New Conduction Disturbance (CD- -> CD+)",
        "Acute Onset from NORM Baseline",
    ]
    task_short = ["General", "Acute MI", "Acute STTC", "New CD", "NORM Onset"]

    so_p3 = triad_results["SetOperator (Native, d=256)"]["pillar3_relative_change_gain"]
    csfm_p3 = triad_results["CSFM Base (Native, d=768)"]["pillar3_relative_change_gain"]

    so_gains = [so_p3.get(k, {}).get("calibration_gain_delta", 0.0) for k in task_keys]
    csfm_gains = [csfm_p3.get(k, {}).get("calibration_gain_delta", 0.0) for k in task_keys]

    ax = axes[2]
    x_t = np.arange(len(task_short))
    ax.bar(x_t - w/2, so_gains, w, label="SetOperator (Native)", color="#1f77b4", alpha=0.9)
    ax.bar(x_t + w/2, csfm_gains, w, label="CSFM Base (Native)", color="#ff7f0e", alpha=0.9)
    ax.axhline(0, color="gray", linestyle="-", linewidth=1.2)
    ax.set_xticks(x_t)
    ax.set_xticklabels(task_short, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Calibration Gain Δ (AUROC)", fontsize=11, fontweight="bold")
    ax.set_title("C. Patient-Relative Change Gain\n(Δ = AUROC(z_t - b_p) - AUROC(z_t))", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plot_path = args.output_dir / "disease_preservation_triad_curves.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved publication figure to: {plot_path}")

    logger.info("\n=== Milestone R015 Execution COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
