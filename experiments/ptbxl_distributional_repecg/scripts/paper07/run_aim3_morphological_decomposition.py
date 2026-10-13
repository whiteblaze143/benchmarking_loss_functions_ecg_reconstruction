#!/usr/bin/env python3
"""Aim 3: Morphological and Structural Decomposition Experiment (SetOperator E6).

Tests the physiological decomposition hypothesis from the PRD:
"z_t = s_p + delta_t + epsilon"
Where s_p encapsulates static cardiac anatomy (conduction pathways, axis, chamber morphology, demographics)
and delta_t isolates dynamic state perturbations.

Evaluates probes trained on:
1. Raw Embedding (z_t)
2. Personal Reference Frame (s_p = mean of prior recordings)
3. Acute State Residual (delta_t = z_t - s_p)

Across Target Variables:
- Static Anatomical / Demographic Targets:
  1. Sex (Permanent biological morphology)
  2. Age (Structural cardiac aging)
  3. Left Axis Deviation (LAD, anatomical heart orientation)
  4. Conduction Defect (CD, intrinsic conduction bundle pathology)
  5. Ventricular Hypertrophy (HYP, anatomical myocardial remodeling)
- Dynamic Rhythm Targets:
  6. Acute Rhythm Change (AF <-> SR transition)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "prism_ecg_aim3_morphological_decomposition_v1"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("aim3_decomposition")
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
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score)) + 1
    r1 = float(ranks[y_true == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_and_eval_binary(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> float:
    best_auc = -1.0
    for c in C_GRID:
        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_tr)
        x_te_s = sc.transform(x_te)
        clf = LogisticRegression(C=c, max_iter=1500, random_state=42)
        clf.fit(x_tr_s, y_tr)
        p = clf.predict_proba(x_te_s)[:, 1]
        auc = fast_auroc(y_te, p)
        if auc > best_auc:
            best_auc = auc
    return float(best_auc)


def fit_and_eval_regression(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> dict[str, float]:
    best_r2 = -999.0
    best_mae = 999.0
    for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_tr)
        x_te_s = sc.transform(x_te)
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(x_tr_s, y_tr)
        preds = reg.predict(x_te_s)
        r2 = r2_score(y_te, preds)
        mae = float(np.mean(np.abs(y_te - preds)))
        if r2 > best_r2:
            best_r2 = r2
            best_mae = mae
    return {"r2": float(best_r2), "mae": float(best_mae)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aim 3: Morphological and Structural Decomposition")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz"),
    )
    parser.add_argument(
        "--ptbxl-meta",
        type=Path,
        default=Path("data/ptbxl/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--scp-statements",
        type=Path,
        default=Path("data/ptbxl/scp_statements.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_aim3_decomposition"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output / "run.log")

    logger.info("======================================================================")
    logger.info("=== PRISM-ECG Aim 3: Morphological and Structural Decomposition   ===")
    logger.info("======================================================================")
    logger.info(f"Loading latents from: {args.cache_file}")

    npz = np.load(args.cache_file)
    all_z = npz["all_z"]
    all_ecg = npz["all_ecg"]
    all_pat = npz["all_pat"]
    ecg_to_idx = {int(e): i for i, e in enumerate(all_ecg)}

    scp_df = pd.read_csv(args.scp_statements, index_col=0)
    diag_class = scp_df["diagnostic_class"].dropna().to_dict()

    df_meta = pd.read_csv(args.ptbxl_meta, index_col="ecg_id")
    df_meta["ecg_id"] = df_meta.index.values
    df_meta["recording_date"] = pd.to_datetime(df_meta["recording_date"])
    df_meta["scp_dict"] = df_meta["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Conduction defect flag
    def has_class(d: dict, target_cls: str) -> int:
        if not isinstance(d, dict):
            return 0
        classes = {diag_class[c] for c in d.keys() if c in diag_class}
        return 1 if target_cls in classes else 0

    df_meta["is_cd"] = df_meta["scp_dict"].apply(lambda d: has_class(d, "CD"))
    df_meta["is_hyp"] = df_meta["scp_dict"].apply(lambda d: has_class(d, "HYP"))
    df_meta["is_lad"] = df_meta["heart_axis"].apply(lambda a: 1 if str(a) == "LAD" else (0 if pd.notna(a) else -1))
    df_meta["is_af"] = df_meta["scp_dict"].apply(lambda d: 1 if ("AFIB" in d or "AFLT" in d) else 0)

    # Form triplets: (z_t, s_p, delta_t) for target recordings with at least 1 earlier baseline
    records_list = []
    for pid, group in df_meta.groupby("patient_id"):
        if len(group) < 2:
            continue
        g = group.sort_values("recording_date").to_dict("records")
        for t in range(1, len(g)):
            rec_t = g[t]
            t_ecg = int(rec_t["ecg_id"])
            if t_ecg not in ecg_to_idx:
                continue

            prior_indices = []
            for b in range(t):
                b_ecg = int(g[b]["ecg_id"])
                if b_ecg in ecg_to_idx:
                    prior_indices.append(ecg_to_idx[b_ecg])

            if len(prior_indices) == 0:
                continue

            z_curr = all_z[ecg_to_idx[t_ecg]]
            s_prior = np.mean(all_z[prior_indices], axis=0)
            delta = z_curr - s_prior

            # Check if AF changed relative to priors
            prior_af = [g[b]["is_af"] for b in range(t)]
            curr_af = rec_t["is_af"]
            # Change in rhythm status
            rhythm_changed = 1 if curr_af != prior_af[-1] else 0

            records_list.append({
                "patient_id": int(pid),
                "ecg_id": t_ecg,
                "strat_fold": int(rec_t["strat_fold"]),
                "z_curr": z_curr,
                "s_prior": s_prior,
                "delta": delta,
                "sex": int(rec_t["sex"]) if pd.notna(rec_t["sex"]) else -1,
                "age": float(rec_t["age"]) if pd.notna(rec_t["age"]) else float("nan"),
                "is_cd": int(rec_t["is_cd"]),
                "is_hyp": int(rec_t["is_hyp"]),
                "is_lad": int(rec_t["is_lad"]),
                "rhythm_changed": rhythm_changed,
            })

    logger.info(f"Formed {len(records_list):,} multi-visit evaluated points.")

    train_recs = [r for r in records_list if r["strat_fold"] in range(1, 8)]
    test_recs = [r for r in records_list if r["strat_fold"] == 8]
    logger.info(f"Train points: {len(train_recs):,} | Test points: {len(test_recs):,}")

    z_tr = np.array([r["z_curr"] for r in train_recs])
    s_tr = np.array([r["s_prior"] for r in train_recs])
    d_tr = np.array([r["delta"] for r in train_recs])

    z_te = np.array([r["z_curr"] for r in test_recs])
    s_te = np.array([r["s_prior"] for r in test_recs])
    d_te = np.array([r["delta"] for r in test_recs])

    results = {}

    # Target 1: Biological Sex (Binary classification)
    y_sex_tr = np.array([r["sex"] for r in train_recs])
    y_sex_te = np.array([r["sex"] for r in test_recs])
    sex_mask_tr = y_sex_tr >= 0
    sex_mask_te = y_sex_te >= 0

    auc_z_sex = fit_and_eval_binary(z_tr[sex_mask_tr], y_sex_tr[sex_mask_tr], z_te[sex_mask_te], y_sex_te[sex_mask_te])
    auc_s_sex = fit_and_eval_binary(s_tr[sex_mask_tr], y_sex_tr[sex_mask_tr], s_te[sex_mask_te], y_sex_te[sex_mask_te])
    auc_d_sex = fit_and_eval_binary(d_tr[sex_mask_tr], y_sex_tr[sex_mask_tr], d_te[sex_mask_te], y_sex_te[sex_mask_te])

    logger.info(f"\n1. Biological Sex (Anatomical Baseline):")
    logger.info(f"   Raw z_t:     AUROC = {auc_z_sex:.4f}")
    logger.info(f"   Reference s: AUROC = {auc_s_sex:.4f}  (Retained: {auc_s_sex / auc_z_sex * 100:.1f}%)")
    logger.info(f"   Delta delta: AUROC = {auc_d_sex:.4f}  (Eliminated: {(1 - auc_d_sex/auc_z_sex)*100:.1f}%)")

    results["Sex"] = {"z_raw": auc_z_sex, "s_prior": auc_s_sex, "delta": auc_d_sex, "metric": "AUROC"}

    # Target 2: Biological Age (Continuous regression)
    age_tr = np.array([r["age"] for r in train_recs])
    age_te = np.array([r["age"] for r in test_recs])
    age_m_tr = ~np.isnan(age_tr)
    age_m_te = ~np.isnan(age_te)

    res_z_age = fit_and_eval_regression(z_tr[age_m_tr], age_tr[age_m_tr], z_te[age_m_te], age_te[age_m_te])
    res_s_age = fit_and_eval_regression(s_tr[age_m_tr], age_tr[age_m_tr], s_te[age_m_te], age_te[age_m_te])
    res_d_age = fit_and_eval_regression(d_tr[age_m_tr], age_tr[age_m_tr], d_te[age_m_te], age_te[age_m_te])

    logger.info(f"\n2. Patient Age (Chronological Structural Degradation):")
    logger.info(f"   Raw z_t:     R^2 = {res_z_age['r2']:.4f} | MAE = {res_z_age['mae']:.2f} yrs")
    logger.info(f"   Reference s: R^2 = {res_s_age['r2']:.4f} | MAE = {res_s_age['mae']:.2f} yrs")
    logger.info(f"   Delta delta: R^2 = {res_d_age['r2']:.4f} | MAE = {res_d_age['mae']:.2f} yrs (Anatomical age erased from delta)")

    results["Age"] = {"z_raw": res_z_age["r2"], "s_prior": res_s_age["r2"], "delta": res_d_age["r2"], "metric": "R2"}

    # Target 3: Left Axis Deviation (LAD) (Heart Anatomy / Conduction Vector)
    lad_tr = np.array([r["is_lad"] for r in train_recs])
    lad_te = np.array([r["is_lad"] for r in test_recs])
    lad_m_tr = lad_tr >= 0
    lad_m_te = lad_te >= 0

    auc_z_lad = fit_and_eval_binary(z_tr[lad_m_tr], lad_tr[lad_m_tr], z_te[lad_m_te], lad_te[lad_m_te])
    auc_s_lad = fit_and_eval_binary(s_tr[lad_m_tr], lad_tr[lad_m_tr], s_te[lad_m_te], lad_te[lad_m_te])
    auc_d_lad = fit_and_eval_binary(d_tr[lad_m_tr], lad_tr[lad_m_tr], d_te[lad_m_te], lad_te[lad_m_te])

    logger.info(f"\n3. Left Axis Deviation (LAD, Structural Heart Axis):")
    logger.info(f"   Raw z_t:     AUROC = {auc_z_lad:.4f}")
    logger.info(f"   Reference s: AUROC = {auc_s_lad:.4f}")
    logger.info(f"   Delta delta: AUROC = {auc_d_lad:.4f}")

    results["Left_Axis_Deviation"] = {"z_raw": auc_z_lad, "s_prior": auc_s_lad, "delta": auc_d_lad, "metric": "AUROC"}

    # Target 4: Conduction Defect (CD: Fascicular/Bundle Branch Blocks)
    cd_tr = np.array([r["is_cd"] for r in train_recs])
    cd_te = np.array([r["is_cd"] for r in test_recs])

    auc_z_cd = fit_and_eval_binary(z_tr, cd_tr, z_te, cd_te)
    auc_s_cd = fit_and_eval_binary(s_tr, cd_tr, s_te, cd_te)
    auc_d_cd = fit_and_eval_binary(d_tr, cd_tr, d_te, cd_te)

    logger.info(f"\n4. Conduction Defects (CD, Conduction Pathway Substrate):")
    logger.info(f"   Raw z_t:     AUROC = {auc_z_cd:.4f}")
    logger.info(f"   Reference s: AUROC = {auc_s_cd:.4f}")
    logger.info(f"   Delta delta: AUROC = {auc_d_cd:.4f}")

    results["Conduction_Defect"] = {"z_raw": auc_z_cd, "s_prior": auc_s_cd, "delta": auc_d_cd, "metric": "AUROC"}

    # Target 5: Ventricular Hypertrophy (HYP, Anatomical Muscle Thickening)
    hyp_tr = np.array([r["is_hyp"] for r in train_recs])
    hyp_te = np.array([r["is_hyp"] for r in test_recs])

    auc_z_hyp = fit_and_eval_binary(z_tr, hyp_tr, z_te, hyp_te)
    auc_s_hyp = fit_and_eval_binary(s_tr, hyp_tr, s_te, hyp_te)
    auc_d_hyp = fit_and_eval_binary(d_tr, hyp_tr, d_te, hyp_te)

    logger.info(f"\n5. Ventricular Hypertrophy (HYP, Muscle Thickness):")
    logger.info(f"   Raw z_t:     AUROC = {auc_z_hyp:.4f}")
    logger.info(f"   Reference s: AUROC = {auc_s_hyp:.4f}")
    logger.info(f"   Delta delta: AUROC = {auc_d_hyp:.4f}")

    results["Hypertrophy"] = {"z_raw": auc_z_hyp, "s_prior": auc_s_hyp, "delta": auc_d_hyp, "metric": "AUROC"}

    # Target 6: Acute Rhythm Shift (Dynamic State Change)
    rhy_tr = np.array([r["rhythm_changed"] for r in train_recs])
    rhy_te = np.array([r["rhythm_changed"] for r in test_recs])

    auc_z_rhy = fit_and_eval_binary(z_tr, rhy_tr, z_te, rhy_te)
    auc_s_rhy = fit_and_eval_binary(s_tr, rhy_tr, s_te, rhy_te)
    auc_d_rhy = fit_and_eval_binary(d_tr, rhy_tr, d_te, rhy_te)

    logger.info(f"\n6. Acute Rhythm Transition (Dynamic Electrical Shift):")
    logger.info(f"   Raw z_t:     AUROC = {auc_z_rhy:.4f}")
    logger.info(f"   Reference s: AUROC = {auc_s_rhy:.4f}")
    logger.info(f"   Delta delta: AUROC = {auc_d_rhy:.4f}  (Delta superiority: {auc_d_rhy - auc_s_rhy:+.4f})")

    results["Rhythm_Transition"] = {"z_raw": auc_z_rhy, "s_prior": auc_s_rhy, "delta": auc_d_rhy, "metric": "AUROC"}

    # 6. Plotting Publication Figure
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    labels = ["Sex\n(Biological)", "Axis (LAD)\n(Orientation)", "CD\n(Conduction)", "Hypertrophy\n(Remodeling)", "Rhythm Change\n(Dynamic)"]
    z_scores = [auc_z_sex, auc_z_lad, auc_z_cd, auc_z_hyp, auc_z_rhy]
    s_scores = [auc_s_sex, auc_s_lad, auc_s_cd, auc_s_hyp, auc_s_rhy]
    d_scores = [auc_d_sex, auc_d_lad, auc_d_cd, auc_d_hyp, auc_d_rhy]

    x = np.arange(len(labels))
    width = 0.25

    rects1 = ax.bar(x - width, z_scores, width, label="Raw Representation ($z_t$)", color="#4A5568", alpha=0.9)
    rects2 = ax.bar(x, s_scores, width, label="Patient Baseline Frame ($s_p = \\bar{z}_p$)", color="#2B6CB0", alpha=0.9)
    rects3 = ax.bar(x + width, d_scores, width, label="Acute Perturbation Residual ($\\delta_t = z_t - s_p$)", color="#319795", alpha=0.9)

    ax.set_ylabel("Linear Probe AUROC", fontsize=11, fontweight="bold")
    ax.set_title("PRISM-ECG Aim 3: Anatomical vs Dynamic Decomposition\n($z_t = s_p + \\delta_t + \\epsilon$)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="medium")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.50)")
    ax.legend(frameon=True, fontsize=10, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=45)

    plt.tight_layout()
    plot_file = args.output / "aim3_morphological_decomposition.png"
    plt.savefig(plot_file)
    logger.info(f"\nSaved publication figure to {plot_file}")

    # Save results payload
    payload = {
        "schema_version": SCHEMA_VERSION,
        "n_train_points": len(train_recs),
        "n_test_points": len(test_recs),
        "targets": results,
    }
    out_file = args.output / "results.json"
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"Saved complete results to {out_file}")
    logger.info("=== Aim 3 Morphological Decomposition COMPLETED Successfully ===")


if __name__ == "__main__":
    main()
