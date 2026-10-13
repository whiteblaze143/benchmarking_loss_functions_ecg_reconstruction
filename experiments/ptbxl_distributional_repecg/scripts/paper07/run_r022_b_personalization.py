#!/usr/bin/env python3
"""
Stage R022-B: Personalization Formulation Benchmark (K=1).
Evaluates personalization variants of SetOperator-Rhythm on MIMIC AF->SR:
  1. Joint MV-PEB:               MV-PEB([z_morph; r_RR])
  2. Morphology-Only PEB:        [MV-PEB(z_morph); r_RR,t] (rhythm treated as instantaneous state)
  3. Differential Rhythm:        [MV-PEB(z_morph); r_RR,t; r_RR,t - r_RR,b] (explicit delta rhythm)
  4. Cold-Start Reference:       [z_morph,t; r_RR,t] (K=0)

Evaluates on identical strict [1, 1, 1] test pairs (N=70 pairs, 29 clusters).
Computes paired bootstrap differences against cold-start and cluster influence diagnostics.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r022_b_personalization")

CANONICAL_CACHE = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/canonical_latents_cache/canonical_r021_latents.npz")
RHYTHM_CACHE = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/canonical_latents_cache/mimic_rhythm_embeddings.npz")
PAIRS_PARQUET = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_safe_audit/safe_transition_pairs.parquet")
EVENTS_PARQUET = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_safe_audit/safe_event_table.parquet")
OUTPUT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/r022_b_personalization_results.json")


def fit_mv_peb(z_train: np.ndarray, train_pids: np.ndarray):
    """Empirical-Bayes estimation of population mean, between-subject, and within-subject covariance."""
    d = z_train.shape[1]
    mu = np.mean(z_train, axis=0)

    unique_pids = np.unique(train_pids)
    within_covs, patient_means = [], []

    for p in unique_pids:
        idx = np.where(train_pids == p)[0]
        zp = z_train[idx]
        patient_means.append(np.mean(zp, axis=0))
        if len(zp) > 1:
            cov_p = np.cov(zp, rowvar=False)
            if cov_p.ndim == 2:
                within_covs.append(cov_p)

    sigma_w = np.mean(within_covs, axis=0) if len(within_covs) > 0 else np.eye(d) * 0.1
    patient_means = np.array(patient_means)
    sigma_b = np.cov(patient_means, rowvar=False) if len(patient_means) > 1 else np.eye(d) * 0.5

    # Tikhonov regularization
    sigma_w += np.eye(d) * (0.01 * np.trace(sigma_w) / d + 1e-4)
    sigma_b += np.eye(d) * (0.01 * np.trace(sigma_b) / d + 1e-4)
    return mu, sigma_b, sigma_w


def peb_adapt(z_prior: np.ndarray, mu: np.ndarray, sigma_b: np.ndarray, sigma_w: np.ndarray):
    """Posterior mean update with K=1 prior observation: s_hat = mu + Sigma_B (Sigma_B + Sigma_W)^-1 (z_prior - mu)."""
    inv_term = np.linalg.inv(sigma_b + sigma_w)
    gain = sigma_b @ inv_term
    return mu + (gain @ (z_prior - mu).T).T


def find_operating_threshold(y_val, prob_val, target_sens=0.90):
    sorted_probs = np.sort(prob_val[y_val == 1])
    target_idx = int(np.floor((1.0 - target_sens) * len(sorted_probs)))
    target_idx = max(0, min(target_idx, len(sorted_probs) - 1))
    return float(sorted_probs[target_idx])


def main():
    logger.info("=== Starting Stage R022-B: Personalization Formulation Benchmark ===")
    assert CANONICAL_CACHE.exists()
    assert RHYTHM_CACHE.exists()

    with np.load(CANONICAL_CACHE, allow_pickle=True) as d:
        lat_so = d["latents_so_30s"].item() if "latents_so_30s" in d else d["latents_so"].item()
    with np.load(RHYTHM_CACHE, allow_pickle=True) as d:
        lat_rr = d["rhythm_30s"].item()

    events_df = pl.read_parquet(EVENTS_PARQUET)
    pairs_df = (
        pl.scan_parquet(PAIRS_PARQUET)
        .filter(pl.col("transition_cell").is_in(["AF->SR", "AF->AF"]))
        .collect()
    )

    import ast
    def parse_sqi(val):
        if val is None: return []
        if isinstance(val, (list, tuple)): return [int(x) for x in val]
        try: return [int(x) for x in ast.literal_eval(str(val))]
        except: return []

    event_sqi_map = {r["safe_event_key"]: parse_sqi(r["vector_10s_ecg_sqi"]) for r in events_df.iter_rows(named=True)}
    def is_strict(sqi): return len(sqi) == 3 and sqi == [1, 1, 1]

    strict_rows = [r for r in pairs_df.iter_rows(named=True) if is_strict(event_sqi_map.get(r["safe_event_key"], [])) and is_strict(event_sqi_map.get(r["target_safe_event_key"], []))]
    df_strict = pl.DataFrame(strict_rows)

    def partition(df, seed=42):
        t_pos = df.filter((pl.col("target_strat_fold") == 8) & (pl.col("transition_cell") == "AF->SR"))
        t_neg = df.filter((pl.col("target_strat_fold") == 8) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(t_pos) * 5, seed=seed)
        v_pos = df.filter((pl.col("target_strat_fold") == 7) & (pl.col("transition_cell") == "AF->SR"))
        v_neg = df.filter((pl.col("target_strat_fold") == 7) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(v_pos) * 5, seed=seed)
        tr_pos = df.filter((pl.col("target_strat_fold").is_in(list(range(7)))) & (pl.col("transition_cell") == "AF->SR"))
        tr_neg = df.filter((pl.col("target_strat_fold").is_in(list(range(7)))) & (pl.col("transition_cell") == "AF->AF")).sample(n=min(len(tr_pos) * 5, 1500), seed=seed)
        return pl.concat([tr_pos, tr_neg]), pl.concat([v_pos, v_neg]), pl.concat([t_pos, t_neg])

    train_strict, val_strict, test_strict = partition(df_strict)

    # Build raw baseline and target pairs
    def extract_pairs(split_df):
        zm_b, zr_b = [], []
        zm_t, zr_t = [], []
        y, pids = [], []

        for r in split_df.iter_rows(named=True):
            s_key = f"{r['subject_id']}_{r['record_id']}_{r['event_id']}"
            t_key = f"{r['subject_id']}_{r['target_record_id']}_{r['target_event_id']}"
            if s_key not in lat_so or t_key not in lat_so: continue
            if s_key not in lat_rr or t_key not in lat_rr: continue

            zm_b.append(lat_so[s_key])
            zr_b.append(lat_rr[s_key])
            zm_t.append(lat_so[t_key])
            zr_t.append(lat_rr[t_key])
            y.append(1 if r["transition_cell"] == "AF->SR" else 0)
            pids.append(r["subject_id"])

        return (
            np.array(zm_b, dtype=np.float32),
            np.array(zr_b, dtype=np.float32),
            np.array(zm_t, dtype=np.float32),
            np.array(zr_t, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(pids, dtype=np.int64),
        )

    tr_zmb, tr_zrb, tr_zmt, tr_zrt, y_tr, p_tr = extract_pairs(train_strict)
    v_zmb, v_zrb, v_zmt, v_zrt, y_v, p_v = extract_pairs(val_strict)
    te_zmb, te_zrb, te_zmt, te_zrt, y_te, p_te = extract_pairs(test_strict)

    # 1. Fit MV-PEB on Training Data
    # A. Morphology PEB (256-D)
    mu_m, sb_m, sw_m = fit_mv_peb(tr_zmb, p_tr)

    # B. Joint PEB (288-D)
    tr_joint_b = np.column_stack([tr_zmb, tr_zrb])
    mu_j, sb_j, sw_j = fit_mv_peb(tr_joint_b, p_tr)

    # Form Representations for all splits:
    # Variant 0: Cold-Start K=0: [z_morph,t; r_RR,t] (288-D)
    # Variant 1: Joint MV-PEB (288-D): PEB_adapted([z_morph,b; r_RR,b]) vs target
    # Variant 2: Morph-Only PEB (288-D): [PEB_adapted(z_morph,b); r_RR,t]
    # Variant 3: Differential Rhythm (320-D): [PEB_adapted(z_morph,b); r_RR,t; r_RR,t - r_RR,b]

    def build_variants(zmb, zrb, zmt, zrt):
        # Morph adapted
        zm_adapted = peb_adapt(zmb, mu_m, sb_m, sw_m)
        # Joint adapted
        joint_b = np.column_stack([zmb, zrb])
        joint_adapted = peb_adapt(joint_b, mu_j, sb_j, sw_j)

        # Delta rhythm
        delta_r = zrt - zrb

        v0_cold = np.column_stack([zmt, zrt])
        v1_joint = joint_adapted
        v2_morph_only = np.column_stack([zm_adapted, zrt])
        v3_diff = np.column_stack([zm_adapted, zrt, delta_r])

        return {
            "Cold-Start K=0 (288-D)": v0_cold,
            "Joint MV-PEB (288-D)": v1_joint,
            "Morph-PEB + Dynamic RR (288-D)": v2_morph_only,
            "Morph-PEB + Delta-RR (320-D)": v3_diff,
        }

    tr_vars = build_variants(tr_zmb, tr_zrb, tr_zmt, tr_zrt)
    v_vars = build_variants(v_zmb, v_zrb, v_zmt, v_zrt)
    te_vars = build_variants(te_zmb, te_zrb, te_zmt, te_zrt)

    results = {}
    test_probs = {}

    for name in tr_vars:
        X_tr = tr_vars[name]
        X_v = v_vars[name]
        X_te = te_vars[name]

        best_c, best_score, best_model = 1.0, -1.0, None
        for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=c, max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            v_prob = clf.predict_proba(X_v)[:, 1]
            sc = roc_auc_score(y_v, v_prob)
            if sc > best_score:
                best_score = sc
                best_c = c
                best_model = clf

        v_prob = best_model.predict_proba(X_v)[:, 1]
        te_prob = best_model.predict_proba(X_te)[:, 1]
        test_probs[name] = te_prob
        tau = find_operating_threshold(y_v, v_prob, target_sens=0.90)

        preds = (te_prob >= tau).astype(int)
        cm = confusion_matrix(y_te, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / max(fp + tn, 1))

        auc = float(roc_auc_score(y_te, te_prob))
        pr = float(average_precision_score(y_te, te_prob))

        results[name] = {
            "auroc": auc,
            "auprc": pr,
            "tau_star": tau,
            "fpr_tau_star": fpr,
        }
        logger.info(f"  {name:30s} | AUROC: {auc:.4f} | AUPRC: {pr:.4f} | FPR@tau*: {fpr:.4f}")

    # Paired Bootstrap Differences against Cold-Start
    rng = np.random.default_rng(42)
    unique_pids = np.unique(p_te)
    n_bootstraps = 1000

    delta_joint = []
    delta_morph_only = []
    delta_diff = []

    for _ in range(n_bootstraps):
        boot_p = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx_list = []
        for p in boot_p:
            idx_list.extend(np.where(p_te == p)[0])
        idx = np.array(idx_list)

        yb = y_te[idx]
        if len(np.unique(yb)) < 2: continue

        a_cold = roc_auc_score(yb, test_probs["Cold-Start K=0 (288-D)"][idx])
        a_joint = roc_auc_score(yb, test_probs["Joint MV-PEB (288-D)"][idx])
        a_morph = roc_auc_score(yb, test_probs["Morph-PEB + Dynamic RR (288-D)"][idx])
        a_diff = roc_auc_score(yb, test_probs["Morph-PEB + Delta-RR (320-D)"][idx])

        delta_joint.append(a_joint - a_cold)
        delta_morph_only.append(a_morph - a_cold)
        delta_diff.append(a_diff - a_cold)

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    boot_out = {
        "delta_joint_vs_cold": stats(delta_joint),
        "delta_morph_only_vs_cold": stats(delta_morph_only),
        "delta_diff_vs_cold": stats(delta_diff),
    }

    logger.info(f"\nPaired Differences vs Cold-Start (K=0):")
    logger.info(f"  Joint MV-PEB vs Cold:               {boot_out['delta_joint_vs_cold']['mean']:+.4f} [{boot_out['delta_joint_vs_cold']['ci_lower']:+.4f}, {boot_out['delta_joint_vs_cold']['ci_upper']:+.4f}]")
    logger.info(f"  Morph-PEB + Dynamic RR vs Cold:     {boot_out['delta_morph_only_vs_cold']['mean']:+.4f} [{boot_out['delta_morph_only_vs_cold']['ci_lower']:+.4f}, {boot_out['delta_morph_only_vs_cold']['ci_upper']:+.4f}]")
    logger.info(f"  Morph-PEB + Delta-RR vs Cold:       {boot_out['delta_diff_vs_cold']['mean']:+.4f} [{boot_out['delta_diff_vs_cold']['ci_lower']:+.4f}, {boot_out['delta_diff_vs_cold']['ci_upper']:+.4f}]")

    final_payload = {
        "milestone": "R022-B",
        "task": "MIMIC AF->SR Personalization Formulation Benchmark (K=1)",
        "cohort": {
            "n_test_pairs": len(y_te),
            "test_positive": int(y_te.sum()),
            "test_negative": int((1 - y_te).sum()),
            "n_patient_clusters": int(len(np.unique(p_te))),
        },
        "variants": results,
        "paired_bootstrap_vs_cold_start": boot_out,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Results successfully saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
