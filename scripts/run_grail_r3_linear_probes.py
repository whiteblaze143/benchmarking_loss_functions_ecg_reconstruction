#!/usr/bin/env python3
"""Leakage-safe deterministic R3 linear-accessibility audit on frozen exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ub", "b0", "b1", "model_001", "model_101", "model_m"]
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
PCA_DIMS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]


def metrics(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    sens95 = float(np.max(tpr[fpr <= .05])) if np.any(fpr <= .05) else 0.0
    spec = 1.0 - fpr
    spec95 = float(np.max(spec[tpr >= .95])) if np.any(tpr >= .95) else 0.0
    return dict(auroc=float(roc_auc_score(y, score)),
                auprc=float(average_precision_score(y, score)),
                sens_at_95spec=sens95, spec_at_95sens=spec95)


def fit_checked(x, y, c):
    clf = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000,
                             tol=1e-8, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not converged:
        clf.set_params(max_iter=20000).fit(x, y)
        converged = bool(np.all(clf.n_iter_ < 20000))
    return clf, converged


def tune_and_fit(z, y, folds, train_mask=None):
    base = np.ones(len(y), dtype=bool) if train_mask is None else train_mask
    fit6 = base & (folds <= 6)
    tune7 = base & (folds == 7)
    final = base & (folds <= 7)
    test = folds == 8
    if min(y[fit6].sum(), len(y[fit6]) - y[fit6].sum(),
           y[tune7].sum(), len(y[tune7]) - y[tune7].sum()) < 1:
        return None
    scaler6 = StandardScaler().fit(z[fit6])
    x6, x7 = scaler6.transform(z[fit6]), scaler6.transform(z[tune7])
    best = None
    for c in CS:
        clf, conv = fit_checked(x6, y[fit6], c)
        if not conv:
            continue
        score = roc_auc_score(y[tune7], clf.predict_proba(x7)[:, 1])
        candidate = (score, -CS.index(c), c)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    scaler = StandardScaler().fit(z[final])
    clf, conv = fit_checked(scaler.transform(z[final]), y[final], best[2])
    if not conv:
        return None
    score = clf.predict_proba(scaler.transform(z[test]))[:, 1]
    return best[2], conv, test, score, metrics(y[test], score)


def concepts(cfg):
    out = [("anchor", c, i) for i, c in enumerate(cfg["all_anchor_codes"])]
    tiers = {c: "P1" for c in cfg["tier_p1_codes"]}
    tiers.update({c: "P2" for c in cfg["tier_p2_codes"]})
    tiers.update({c: "P3" for c in cfg["tier_p3_codes"]})
    out += [(tiers[c], c, i) for i, c in enumerate(cfg["all_probe_codes"])]
    return out


def load_model(root, model):
    z = np.load(root / model / "embeddings_folds1_8.npz")
    return {k: z[k] for k in z.files}


def shared_patient_masks(patient, folds, repeats=20):
    rng = np.random.default_rng(20260911)
    patients = np.unique(patient[folds <= 6])
    masks = {}
    for frac in FRACTIONS:
        for repeat in range(repeats):
            n = max(1, round(frac * len(patients)))
            chosen = patients if frac == 1 else rng.choice(patients, n, replace=False)
            masks[(frac, repeat)] = np.isin(patient, chosen) | (folds == 7)
    return masks


def main():
    raise RuntimeError(
        "Retired unsafe monolithic scheduler. Use run_grail_r3_primary_registry.py "
        "followed by run_grail_r3_geometry_residual.py and "
        "run_grail_r3_frozen_c_followups.py."
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-dir", type=Path, default=ROOT / "results/grail_v2/r3_linear_probe_audit/embeddings")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/grail_v2/r3_linear_probe_audit")
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/ptbxl_concept_tiers.yaml").read_text())
    concept_list = concepts(cfg)
    ref = load_model(args.embedding_dir, MODELS[0])
    folds, patients = ref["strat_fold"], ref["patient_id"]
    masks = shared_patient_masks(patients, folds)
    primary, pca_rows, low_rows = [], [], []
    for model in MODELS:
        print(f"R3 probes: {model}", flush=True)
        data = load_model(args.embedding_dir, model); z = data["z"]
        if not (np.array_equal(data["ecg_id"], ref["ecg_id"]) and
                np.array_equal(data["patient_id"], patients)):
            raise RuntimeError(f"record ordering mismatch for {model}")
        for tier, code, idx in concept_list:
            y = data["anchor_labels"][:, idx].astype(int) if tier == "anchor" else data["probe_labels"][:, idx].astype(int)
            result = tune_and_fit(z, y, folds)
            if result is None:
                primary.append(dict(model=model, concept=code, tier=tier, converged=False))
                continue
            c, conv, test, score, met = result
            primary.append(dict(model=model, concept=code, tier=tier,
                                n_positive=int(y[test].sum()), n_negative=int((1-y[test]).sum()),
                                selected_C=c, converged=conv, **met))

            dims = PCA_DIMS + ([128] if model == "ub" else [])
            for k in dims:
                if k > z.shape[1]: continue
                pfit = folds <= 7
                scaler = StandardScaler().fit(z[pfit])
                zs = scaler.transform(z)
                pca = PCA(n_components=k, svd_solver="full").fit(zs[pfit])
                pres = tune_and_fit(pca.transform(zs), y, folds)
                if pres is not None:
                    pca_rows.append(dict(model=model, concept=code, tier=tier, k=k,
                                         selected_C=pres[0], converged=pres[1], **pres[4]))

            for (frac, repeat), mask in masks.items():
                eligible = mask & (folds <= 6)
                if min(y[eligible].sum(), len(y[eligible])-y[eligible].sum()) < 5:
                    continue
                low = tune_and_fit(z, y, folds, mask)
                if low is not None:
                    low_rows.append(dict(model=model, concept=code, tier=tier,
                                         fraction=frac, repeat=repeat,
                                         n_train=int(eligible.sum()), selected_C=low[0],
                                         converged=low[1], **low[4]))

        pd.DataFrame(primary).to_parquet(args.output_dir / "per_concept_linear_probes.partial.parquet", index=False)
        pd.DataFrame(pca_rows).to_parquet(args.output_dir / "pca_clinical_dimensionality.partial.parquet", index=False)
        pd.DataFrame(low_rows).to_parquet(args.output_dir / "low_shot_shared_patient.partial.parquet", index=False)

    primary_df = pd.DataFrame(primary)
    ub = primary_df[primary_df.model == "ub"][["concept", "tier", "auroc"]].rename(columns={"auroc":"ub_auroc"})
    primary_df = primary_df.merge(ub, on=["concept", "tier"], how="left")
    primary_df["ub_ratio_stable"] = primary_df.ub_auroc > .55
    primary_df["ub_normalized_accessibility"] = np.where(
        primary_df.ub_ratio_stable,
        (primary_df.auroc - .5) / (primary_df.ub_auroc - .5), np.nan)
    primary_df.to_parquet(args.output_dir / "per_concept_linear_probes.parquet", index=False)

    pca_df = pd.DataFrame(pca_rows)
    full = primary_df[["model","concept","tier","auroc"]].rename(columns={"auroc":"full_auroc"})
    pca_df = pca_df.merge(full, on=["model","concept","tier"], how="left")
    pca_df["chance_adjusted_retention"] = (pca_df.auroc-.5)/(pca_df.full_auroc-.5)
    pca_df.to_parquet(args.output_dir / "pca_clinical_dimensionality.parquet", index=False)
    d95 = pca_df[pca_df.chance_adjusted_retention >= .95].groupby(["model","concept","tier"], as_index=False).k.min()
    d95.to_parquet(args.output_dir / "clinical_d95.parquet", index=False)

    low_df = pd.DataFrame(low_rows)
    low_df.to_parquet(args.output_dir / "low_shot_shared_patient.parquet", index=False)
    summary = low_df.groupby(["model","tier","fraction"], as_index=False).auroc.agg(["mean","std","count"]).reset_index()
    summary["ci95_half_width"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary.to_csv(args.output_dir / "low_shot_summary.csv", index=False)
    manifest = {"status":"complete", "models":MODELS, "fit_folds":[1,2,3,4,5,6],
                "tuning_fold":7, "evaluation_fold":8, "C_grid":CS,
                "low_shot_repeats":20, "patient_level_subsamples":True,
                "fold8_used_for_selection":False}
    (args.output_dir / "linear_probe_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("R3 linear-accessibility audit complete")


if __name__ == "__main__": main()
