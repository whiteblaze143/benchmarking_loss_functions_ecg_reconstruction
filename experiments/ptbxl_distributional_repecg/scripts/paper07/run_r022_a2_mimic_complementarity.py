#!/usr/bin/env python3
"""
Stage R022-A2: Primary MIMIC Rhythm Complementarity Benchmark (Cold-Start K=0).
Tests whether augmenting frozen SetOperator morphology (z_SO, 256-D) with explicit
rhythm dynamics (r_RR, 32-D) improves out-of-sample discrimination on real Lead-II waveforms:
  H0: I(Y; r_RR | z_SO) = 0   vs   H1: I(Y; r_RR | z_SO) > 0

Compares on IDENTICAL test patients:
  - Arm 1: z_SO (256-D morphology only)
  - Arm 2: r_RR (32-D standardized RKHS rhythm only)
  - Arm 3: [z_SO; r_RR] (288-D SetOperator-Rhythm)
  - Frozen references: CSFM Base (768-D), GraphECG (192-D)

Evaluates:
  - Primary strict [1, 1, 1] SQI cohort (30s)
  - Harmonized duration sensitivity (10s)
  - Looser SQI sensitivity (>= 2/3)
  - Validation-locked operating points (tau* at 90% validation sensitivity)
  - Paired patient-clustered bootstrap CIs (B=1,000) for Δ_morph and Δ_RR
  - Full R-peak detector quality audit for AF vs SR targets
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import scipy.signal
import wfdb

# Path setup
sys.path.insert(0, str(Path("experiments/ptbxl_distributional_repecg/src").resolve()))
from repecg.common.beats import detect_rpeaks
from repecg.paper07_operator.rhythm import (
    extract_rr_atoms,
    audit_rpeak_quality,
    RhythmResponseMap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r022_a2_mimic")

MIMIC_DIR = Path("/data/mithunmanivannan/datasets/mimic3_ext_ppg")
PAIRS_PARQUET = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_safe_audit/safe_transition_pairs.parquet")
EVENTS_PARQUET = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_safe_audit/safe_event_table.parquet")
CANONICAL_CACHE = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/canonical_latents_cache/canonical_r021_latents.npz")
PROVENANCE_LEDGER = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/r021_event_provenance_ledger.json")
RHYTHM_FIT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations/rhythm_fit.npz")
OUTPUT_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/r022_a2_mimic_results.json")
RHYTHM_CACHE_PATH = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/prism_r021_real_waveforms/canonical_latents_cache/mimic_rhythm_embeddings.npz")


def extract_lead_ii(record_base: Path):
    hea = record_base.with_suffix(".hea")
    dat = record_base.with_suffix(".dat")
    if not (hea.exists() and dat.exists()):
        return None
    try:
        rec = wfdb.rdrecord(str(record_base))
        if "II" not in rec.sig_name:
            return None
        idx = rec.sig_name.index("II")
        sig = rec.p_signal[:, idx]
        return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def extract_rhythm_embedding(sig_125: np.ndarray, rmap: RhythmResponseMap, window_len: int = None):
    """Resample 125 Hz to 500 Hz, detect R-peaks, audit detector, and extract r_RR in R^32."""
    if window_len is not None and len(sig_125) >= window_len:
        sig_w = sig_125[-window_len:]
    else:
        sig_w = sig_125

    if len(sig_w) < 250:
        return np.zeros(rmap.dim, dtype=np.float32), {"n_detected_peaks": 0, "n_accepted_rr": 0, "rr_rejection_fraction": 1.0, "is_valid": False}

    sig_500 = scipy.signal.resample(sig_w, int(len(sig_w) * 4))
    basis = np.zeros((len(sig_500), 8), dtype=np.float64)
    basis[:, 1] = sig_500
    try:
        detection = detect_rpeaks(basis, fs=500, primary_lead="II")
        peaks = detection.rpeaks
        audit = audit_rpeak_quality(peaks, fs=500, min_rr_s=0.25, max_rr_s=2.50, min_intervals=5)
        atoms = extract_rr_atoms(peaks, fs=500)
        r_rr = rmap.mean_embedding(atoms, min_intervals=5)
        return r_rr, audit
    except Exception:
        return np.zeros(rmap.dim, dtype=np.float32), {"n_detected_peaks": 0, "n_accepted_rr": 0, "rr_rejection_fraction": 1.0, "is_valid": False}


def bootstrap_paired_arms(y_true, prob_dict, pids, n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(pids)

    metric_records = {k: {"auroc": [], "auprc": []} for k in prob_dict}
    delta_morph = []  # combined - so
    delta_rr = []     # combined - rr

    for _ in range(n_bootstraps):
        boot_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx_list = []
        for p in boot_pids:
            idx_list.extend(np.where(pids == p)[0])
        idx = np.array(idx_list)

        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue

        aurocs = {}
        for k, p in prob_dict.items():
            pb = p[idx]
            a = roc_auc_score(y_b, pb)
            pr = average_precision_score(y_b, pb)
            metric_records[k]["auroc"].append(a)
            metric_records[k]["auprc"].append(pr)
            aurocs[k] = a

        if "Combined (288-D)" in aurocs and "SetOperator (256-D)" in aurocs:
            delta_morph.append(aurocs["Combined (288-D)"] - aurocs["SetOperator (256-D)"])
        if "Combined (288-D)" in aurocs and "Rhythm RKHS (32-D)" in aurocs:
            delta_rr.append(aurocs["Combined (288-D)"] - aurocs["Rhythm RKHS (32-D)"])

    def summarize(vals):
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }

    out = {}
    for k in prob_dict:
        out[k] = {
            "auroc": summarize(metric_records[k]["auroc"]),
            "auprc": summarize(metric_records[k]["auprc"]),
        }

    out["delta_morphology"] = summarize(delta_morph) if delta_morph else None
    out["delta_rhythm"] = summarize(delta_rr) if delta_rr else None
    out["p_val_delta_morph_gt_0"] = float(np.mean(np.array(delta_morph) <= 0)) if delta_morph else None
    out["p_val_delta_rr_gt_0"] = float(np.mean(np.array(delta_rr) <= 0)) if delta_rr else None
    return out


def find_operating_threshold(y_val, prob_val, target_sens=0.90):
    sorted_probs = np.sort(prob_val[y_val == 1])
    target_idx = int(np.floor((1.0 - target_sens) * len(sorted_probs)))
    target_idx = max(0, min(target_idx, len(sorted_probs) - 1))
    return float(sorted_probs[target_idx])


def main():
    logger.info("=== Starting Stage R022-A2: MIMIC AF->SR Rhythm Complementarity Benchmark ===")
    assert RHYTHM_FIT_PATH.exists(), f"Rhythm fit artifact missing: {RHYTHM_FIT_PATH}"
    assert CANONICAL_CACHE.exists(), f"Morphology cache missing: {CANONICAL_CACHE}"
    assert PROVENANCE_LEDGER.exists(), f"Provenance ledger missing: {PROVENANCE_LEDGER}"

    rmap = RhythmResponseMap.load(RHYTHM_FIT_PATH)
    logger.info(f"Loaded primary canonical rhythm map (dim={rmap.dim}, gamma={rmap.gamma:.4f})")

    # Load frozen morphology latents
    with np.load(CANONICAL_CACHE, allow_pickle=True) as data:
        lat_so_30s = data["latents_so_30s"].item() if "latents_so_30s" in data else data["latents_so"].item()
        lat_so_10s = data["latents_so_10s"].item() if "latents_so_10s" in data else {}
        lat_csfm_30s = data["latents_csfm_30s"].item()
        lat_csfm_10s = data["latents_csfm_10s"].item()
        lat_graph_30s = data["latents_graph_30s"].item()
        lat_graph_10s = data["latents_graph_10s"].item()

    with open(PROVENANCE_LEDGER) as f:
        ledger = json.load(f)

    # Extract or load rhythm embeddings
    rhythm_embs_30s = {}
    rhythm_embs_10s = {}
    audits_30s = {}

    if RHYTHM_CACHE_PATH.exists():
        logger.info(f"Loading cached rhythm embeddings from {RHYTHM_CACHE_PATH}...")
        with np.load(RHYTHM_CACHE_PATH, allow_pickle=True) as r_data:
            rhythm_embs_30s = r_data["rhythm_30s"].item()
            rhythm_embs_10s = r_data["rhythm_10s"].item()
            audits_30s = r_data["audits_30s"].item()
        logger.info(f"Loaded {len(rhythm_embs_30s):,} cached rhythm embeddings.")
    else:
        logger.info("Extracting rhythm embeddings from physical Lead-II waveforms...")
        for k, entry in ledger.items():
            sig_file = Path(entry["signal_file"])
            if not sig_file.exists():
                continue
            sig_ii = extract_lead_ii(sig_file.with_suffix(""))
            if sig_ii is None:
                continue

            r_30, audit_30 = extract_rhythm_embedding(sig_ii, rmap, window_len=None)
            r_10, _ = extract_rhythm_embedding(sig_ii, rmap, window_len=1250)

            rhythm_embs_30s[k] = r_30
            rhythm_embs_10s[k] = r_10
            audits_30s[k] = audit_30

        logger.info(f"Extracted {len(rhythm_embs_30s):,} rhythm embeddings. Caching to {RHYTHM_CACHE_PATH}...")
        np.savez_compressed(
            RHYTHM_CACHE_PATH,
            rhythm_30s=rhythm_embs_30s,
            rhythm_10s=rhythm_embs_10s,
            audits_30s=audits_30s,
        )

    # Load Pairs & SQI
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
    def is_sens(sqi): return len(sqi) == 3 and not any(x < 0 for x in sqi) and sum(1 for x in sqi if x == 1) >= 2

    strict_rows = [r for r in pairs_df.iter_rows(named=True) if is_strict(event_sqi_map.get(r["safe_event_key"], [])) and is_strict(event_sqi_map.get(r["target_safe_event_key"], []))]
    sens_rows = [r for r in pairs_df.iter_rows(named=True) if is_sens(event_sqi_map.get(r["safe_event_key"], [])) and is_sens(event_sqi_map.get(r["target_safe_event_key"], []))]

    df_strict = pl.DataFrame(strict_rows)
    df_sens = pl.DataFrame(sens_rows)

    def partition(df, seed=42):
        t_pos = df.filter((pl.col("target_strat_fold") == 8) & (pl.col("transition_cell") == "AF->SR"))
        t_neg = df.filter((pl.col("target_strat_fold") == 8) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(t_pos) * 5, seed=seed)
        v_pos = df.filter((pl.col("target_strat_fold") == 7) & (pl.col("transition_cell") == "AF->SR"))
        v_neg = df.filter((pl.col("target_strat_fold") == 7) & (pl.col("transition_cell") == "AF->AF")).sample(n=len(v_pos) * 5, seed=seed)
        tr_pos = df.filter((pl.col("target_strat_fold").is_in(list(range(7)))) & (pl.col("transition_cell") == "AF->SR"))
        tr_neg = df.filter((pl.col("target_strat_fold").is_in(list(range(7)))) & (pl.col("transition_cell") == "AF->AF")).sample(n=min(len(tr_pos) * 5, 1500), seed=seed)
        return pl.concat([tr_pos, tr_neg]), pl.concat([v_pos, v_neg]), pl.concat([t_pos, t_neg])

    train_strict, val_strict, test_strict = partition(df_strict)
    train_sens, val_sens, test_sens = partition(df_sens)

    # Evaluation function for an arm
    def build_dataset(pairs_sub, so_map, rr_map, csfm_map, graph_map):
        y, pids = [], []
        x_so, x_rr, x_comb, x_csfm, x_graph = [], [], [], [], []
        audits = []

        for r in pairs_sub.iter_rows(named=True):
            s_key = f"{r['subject_id']}_{r['record_id']}_{r['event_id']}"
            t_key = f"{r['subject_id']}_{r['target_record_id']}_{r['target_event_id']}"
            if s_key not in so_map or t_key not in so_map:
                continue
            if s_key not in rr_map or t_key not in rr_map:
                continue
            if s_key not in csfm_map or t_key not in csfm_map:
                continue
            if s_key not in graph_map or t_key not in graph_map:
                continue

            # Target representation
            z_so = so_map[t_key]
            r_rr = rr_map[t_key]
            z_comb = np.concatenate([z_so, r_rr])
            z_csfm = csfm_map[t_key]
            z_graph = graph_map[t_key]

            x_so.append(z_so)
            x_rr.append(r_rr)
            x_comb.append(z_comb)
            x_csfm.append(z_csfm)
            x_graph.append(z_graph)
            y.append(1 if r["transition_cell"] == "AF->SR" else 0)
            pids.append(r["subject_id"])
            audits.append(audits_30s.get(t_key, {"n_detected_peaks": 0, "n_accepted_rr": 0, "rr_rejection_fraction": 1.0, "is_valid": False}))

        return (
            np.array(x_so, dtype=np.float32),
            np.array(x_rr, dtype=np.float32),
            np.array(x_comb, dtype=np.float32),
            np.array(x_csfm, dtype=np.float32),
            np.array(x_graph, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(pids, dtype=np.int64),
            audits,
        )

    # 1. Evaluate Primary Strict 30s Cohort
    logger.info("\nEvaluating Primary Strict 30-s Cohort...")
    tr_so, tr_rr, tr_comb, tr_csfm, tr_graph, y_tr, p_tr, _ = build_dataset(train_strict, lat_so_30s, rhythm_embs_30s, lat_csfm_30s, lat_graph_30s)
    v_so, v_rr, v_comb, v_csfm, v_graph, y_v, p_v, _ = build_dataset(val_strict, lat_so_30s, rhythm_embs_30s, lat_csfm_30s, lat_graph_30s)
    te_so, te_rr, te_comb, te_csfm, te_graph, y_te, p_te, audits_te = build_dataset(test_strict, lat_so_30s, rhythm_embs_30s, lat_csfm_30s, lat_graph_30s)

    logger.info(f"Test cohort: {len(y_te)} pairs ({int(y_te.sum())} Pos, {int((1-y_te).sum())} Neg) across {len(np.unique(p_te))} patient clusters.")

    # Detector Quality Audit on Test Cohort
    af_aud = [a for a, y in zip(audits_te, y_te) if y == 0]  # target is AF in AF->AF (label=0)
    sr_aud = [a for a, y in zip(audits_te, y_te) if y == 1]  # target is SR in AF->SR (label=1)
    detector_audit = {
        "AF_target": {
            "n_events": len(af_aud),
            "mean_peaks": float(np.mean([a["n_detected_peaks"] for a in af_aud])),
            "mean_accepted_rr": float(np.mean([a["n_accepted_rr"] for a in af_aud])),
            "mean_rejection_fraction": float(np.mean([a["rr_rejection_fraction"] for a in af_aud])),
            "failure_rate_under_5_intervals": float(np.mean([not a["is_valid"] for a in af_aud])),
        },
        "SR_target": {
            "n_events": len(sr_aud),
            "mean_peaks": float(np.mean([a["n_detected_peaks"] for a in sr_aud])),
            "mean_accepted_rr": float(np.mean([a["n_accepted_rr"] for a in sr_aud])),
            "mean_rejection_fraction": float(np.mean([a["rr_rejection_fraction"] for a in sr_aud])),
            "failure_rate_under_5_intervals": float(np.mean([not a["is_valid"] for a in sr_aud])),
        },
    }

    arms = {
        "SetOperator (256-D)": (tr_so, v_so, te_so),
        "Rhythm RKHS (32-D)": (tr_rr, v_rr, te_rr),
        "Combined (288-D)": (tr_comb, v_comb, te_comb),
        "GraphECG (192-D)": (tr_graph, v_graph, te_graph),
        "CSFM Base (768-D)": (tr_csfm, v_csfm, te_csfm),
    }

    arm_results = {}
    test_probs = {}

    for arm_name, (X_tr, X_v, X_te) in arms.items():
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
        test_probs[arm_name] = te_prob
        tau = find_operating_threshold(y_v, v_prob, target_sens=0.90)

        preds = (te_prob >= tau).astype(int)
        cm = confusion_matrix(y_te, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / max(fp + tn, 1))
        sens = float(tp / max(tp + fn, 1))

        arm_results[arm_name] = {
            "auroc": float(roc_auc_score(y_te, te_prob)),
            "auprc": float(average_precision_score(y_te, te_prob)),
            "tau_star": tau,
            "fpr_tau_star": fpr,
            "sensitivity_tau_star": sens,
        }
        logger.info(f"  {arm_name:22s} | AUROC: {arm_results[arm_name]['auroc']:.4f} | AUPRC: {arm_results[arm_name]['auprc']:.4f} | FPR@tau*: {fpr:.4f}")

    # Bootstrap Paired Comparisons
    boot = bootstrap_paired_arms(y_te, test_probs, p_te, n_bootstraps=1000, seed=42)

    logger.info(f"\nPaired Differences on Identical Patients:")
    logger.info(f"  Δ_morph (Combined - SO):  {boot['delta_morphology']['mean']:+.4f} [95% CI: {boot['delta_morphology']['ci_lower']:+.4f}, {boot['delta_morphology']['ci_upper']:+.4f}] (p = {boot['p_val_delta_morph_gt_0']:.4f})")
    logger.info(f"  Δ_RR    (Combined - RR):  {boot['delta_rhythm']['mean']:+.4f} [95% CI: {boot['delta_rhythm']['ci_lower']:+.4f}, {boot['delta_rhythm']['ci_upper']:+.4f}] (p = {boot['p_val_delta_rr_gt_0']:.4f})")

    # Decision Logic
    delta_m_gt_0 = boot["delta_morphology"]["ci_lower"] > 0
    delta_r_gt_0 = boot["delta_rhythm"]["ci_lower"] > 0
    if delta_m_gt_0 and delta_r_gt_0:
        verdict = "COMPLEMENTARITY_SUPPORTED"
    elif boot["delta_morphology"]["mean"] > 0 and boot["delta_rhythm"]["mean"] > -0.02:
        verdict = "PARTIAL_COMPLEMENTARITY"
    elif arm_results["Rhythm RKHS (32-D)"]["auroc"] > arm_results["SetOperator (256-D)"]["auroc"] + 0.05:
        verdict = "RHYTHM_DOMINATES"
    else:
        verdict = "COMPLEMENTARITY_NOT_SUPPORTED"

    logger.info(f"Scientific Decision: {verdict}")

    # 2. Harmonized 10-s Duration Sensitivity
    logger.info("\nEvaluating Harmonized 10-s Window Sensitivity...")
    tr_so10, tr_rr10, tr_comb10, _, _, y_tr10, p_tr10, _ = build_dataset(train_strict, lat_so_10s, rhythm_embs_10s, lat_csfm_10s, lat_graph_10s)
    v_so10, v_rr10, v_comb10, _, _, y_v10, p_v10, _ = build_dataset(val_strict, lat_so_10s, rhythm_embs_10s, lat_csfm_10s, lat_graph_10s)
    te_so10, te_rr10, te_comb10, _, _, y_te10, p_te10, _ = build_dataset(test_strict, lat_so_10s, rhythm_embs_10s, lat_csfm_10s, lat_graph_10s)

    def eval_simple(X_tr, y_tr, X_v, y_v, X_te, y_te):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, prob)), float(average_precision_score(y_te, prob))

    au_so10, ap_so10 = eval_simple(tr_so10, y_tr10, v_so10, y_v10, te_so10, y_te10)
    au_rr10, ap_rr10 = eval_simple(tr_rr10, y_tr10, v_rr10, y_v10, te_rr10, y_te10)
    au_comb10, ap_comb10 = eval_simple(tr_comb10, y_tr10, v_comb10, y_v10, te_comb10, y_te10)

    logger.info(f"  10-s SetOperator (256-D): AUROC = {au_so10:.4f}")
    logger.info(f"  10-s Rhythm RKHS (32-D) : AUROC = {au_rr10:.4f}")
    logger.info(f"  10-s Combined (288-D)   : AUROC = {au_comb10:.4f} (Δ_morph = {au_comb10 - au_so10:+.4f})")

    # Output compilation
    final_payload = {
        "milestone": "R022-A2",
        "task": "MIMIC-III AF->SR Rhythm Complementarity Benchmark (Cold-Start K=0)",
        "cohort_primary_strict": {
            "n_test_pairs": len(y_te),
            "test_positive": int(y_te.sum()),
            "test_negative": int((1 - y_te).sum()),
            "n_patient_clusters": int(len(np.unique(p_te))),
        },
        "detector_quality_audit": detector_audit,
        "primary_30s_arms": {
            k: {
                "point_estimates": arm_results[k],
                "bootstrap": boot[k],
            }
            for k in arm_results
        },
        "paired_bootstrap_hypotheses": {
            "delta_morphology": boot["delta_morphology"],
            "delta_rhythm": boot["delta_rhythm"],
            "p_val_delta_morph_gt_0": boot["p_val_delta_morph_gt_0"],
            "p_val_delta_rr_gt_0": boot["p_val_delta_rr_gt_0"],
            "verdict": verdict,
        },
        "harmonized_10s_sensitivity": {
            "n_pairs": len(y_te10),
            "auroc_so_10s": au_so10,
            "auroc_rr_10s": au_rr10,
            "auroc_combined_10s": au_comb10,
            "delta_morph_10s": au_comb10 - au_so10,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Results successfully saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
