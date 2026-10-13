#!/usr/bin/env python3
"""
Stage R022-C: General-Purpose No-Harm Gate.
Verifies that augmenting SetOperator with the 32-D Rhythm channel does not harm
its general-purpose diagnostic capability on PTB-XL superdiagnostic tasks (NORM, MI, STTC, CD, HYP).
Criterion: Macro AUROC drop <= 0.005.
"""

import sys
import json
import ast
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import wfdb

# Path setup
sys.path.insert(0, str(Path("experiments/ptbxl_distributional_repecg/src").resolve()))
from repecg.common.beats import detect_rpeaks
from repecg.paper07_operator.rhythm import extract_rr_atoms, RhythmResponseMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r022_c_noharm")

PTBXL_DIR = Path("data/ptb_xl")
SO_CACHE = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_cache/setoperator_e6_all_z.npz")
RHYTHM_FIT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations/rhythm_fit.npz")
OUTPUT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/r022_c_noharm_results.json")
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def main():
    logger.info("=== Starting Stage R022-C: General-Purpose No-Harm Gate ===")
    assert SO_CACHE.exists(), f"SetOperator cache missing: {SO_CACHE}"
    assert RHYTHM_FIT_PATH.exists(), f"Rhythm fit missing: {RHYTHM_FIT_PATH}"

    rmap = RhythmResponseMap.load(RHYTHM_FIT_PATH)
    logger.info("Loaded canonical RhythmResponseMap.")

    # 1. Load Metadata & Statements
    scp_df = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)
    diag_map = scp_df["diagnostic_class"].dropna().to_dict()

    meta = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    for sc in SUPERCLASSES:
        meta[sc] = meta["scp_codes"].apply(
            lambda s: int(any(diag_map.get(k) == sc for k in ast.literal_eval(s).keys()))
            if isinstance(s, str) else 0
        )

    # 2. Load precomputed morphology latents
    so_npz = np.load(SO_CACHE)
    all_z_so = so_npz["all_z"]
    all_ecg_so = so_npz["all_ecg"]
    ecg_to_idx = {int(e): i for i, e in enumerate(all_ecg_so)}

    # Subsample balanced evaluation set from Folds 1-7 (train, N=2000) and Fold 8 (test, N=1000)
    dev_ecgs = [e for e in all_ecg_so if meta.loc[e, "strat_fold"] in range(1, 8)]
    test_ecgs = [e for e in all_ecg_so if meta.loc[e, "strat_fold"] == 8]

    rng = np.random.default_rng(42)
    sub_dev = rng.choice(dev_ecgs, size=min(len(dev_ecgs), 2000), replace=False)
    sub_test = rng.choice(test_ecgs, size=min(len(test_ecgs), 1000), replace=False)

    eval_ecgs = np.concatenate([sub_dev, sub_test])
    logger.info(f"Extracting rhythm embeddings for {len(eval_ecgs)} records (2,000 train, 1,000 test)...")

    r_embs = []
    so_embs = []
    valid_ecgs = []

    for idx, ecg_id in enumerate(eval_ecgs):
        row = meta.loc[ecg_id]
        rec_path = PTBXL_DIR / row["filename_hr"]
        if not rec_path.with_suffix(".hea").exists():
            continue
        try:
            rec = wfdb.rdrecord(str(rec_path))
            if "II" not in rec.sig_name:
                continue
            sig = np.nan_to_num(rec.p_signal[:, rec.sig_name.index("II")], nan=0.0)
            basis = np.zeros((len(sig), 8))
            basis[:, 1] = sig
            det = detect_rpeaks(basis, fs=500, primary_lead="II")
            atoms = extract_rr_atoms(det.rpeaks, fs=500)
            remb = rmap.mean_embedding(atoms, min_intervals=5)

            z_so = all_z_so[ecg_to_idx[ecg_id]]
            so_embs.append(z_so)
            r_embs.append(remb)
            valid_ecgs.append(ecg_id)
        except Exception:
            continue

    so_arr = np.array(so_embs, dtype=np.float32)
    rr_arr = np.array(r_embs, dtype=np.float32)
    comb_arr = np.column_stack([so_arr, rr_arr])
    sub_meta = meta.loc[valid_ecgs]

    train_mask = sub_meta["strat_fold"].isin(range(1, 8)).values
    test_mask = (sub_meta["strat_fold"] == 8).values

    logger.info(f"Final valid dataset: {train_mask.sum()} train, {test_mask.sum()} test records.")

    def evaluate_rep(X, name):
        X_tr = X[train_mask]
        X_te = X[test_mask]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        class_aucs = {}
        for sc in SUPERCLASSES:
            y_tr = sub_meta[sc].values[train_mask]
            y_te = sub_meta[sc].values[test_mask]
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            clf.fit(X_tr_s, y_tr)
            probs = clf.predict_proba(X_te_s)[:, 1]
            auc = float(roc_auc_score(y_te, probs))
            class_aucs[sc] = auc

        macro = float(np.mean(list(class_aucs.values())))
        logger.info(f"  {name:25s} | Macro AUROC: {macro:.4f} | Per-class: {class_aucs}")
        return {"macro_auroc": macro, "per_class_auroc": class_aucs}

    logger.info("\nEvaluating Superdiagnostic Linear Probes...")
    res_so = evaluate_rep(so_arr, "SetOperator (256-D)")
    res_comb = evaluate_rep(comb_arr, "SetOperator-Rhythm (288-D)")

    macro_so = res_so["macro_auroc"]
    macro_comb = res_comb["macro_auroc"]
    delta_macro = macro_comb - macro_so

    no_harm_passed = bool(delta_macro >= -0.005)
    logger.info(f"\nNo-Harm Gate Audit: ΔMacro AUROC = {delta_macro:+.4f} (Criterion: >= -0.005) -> {'PASSED' if no_harm_passed else 'FAILED'}")

    final_payload = {
        "milestone": "R022-C",
        "task": "PTB-XL Superdiagnostic Preservation No-Harm Gate",
        "cohort": {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "superclasses": SUPERCLASSES,
        },
        "arm_setoperator_256d": res_so,
        "arm_setoperator_rhythm_288d": res_comb,
        "delta_macro_auroc": delta_macro,
        "no_harm_gate_passed": no_harm_passed,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Results successfully saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
