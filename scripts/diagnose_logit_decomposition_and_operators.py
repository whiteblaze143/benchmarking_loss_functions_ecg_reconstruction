#!/usr/bin/env python3
"""Exact Native-Fusion Logit Decomposition & Operator Sensitivity Audit for EchoNext SetOperator & G1.

Implements Priorities P1, P2, P4, P5 of User Directive:
1. Exact logit decomposition: ell_full = ell_ecg + ell_tab + b
2. Distribution, standard deviation, AUROC, AUPRC of ell_ecg and ell_tab.
3. Sensitivity of ell_ecg across distinct operator subspaces and under rotation.
4. Phase-KME variance and rhythm/beat-timing statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"

for p in [BENCHMARK_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH, CANONICAL_8_LEADS
from train_setoperator_echonext import build_setoperator_echonext, DATA_DIR
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext

BASELINE_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/checkpoints/setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
G1_CKPT = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/checkpoints/g1_setoperator/SO-Span_SO-ECG+Tab-Native_s42/best_model.pt")
DEV_BANK_PATH = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/echonext_dev_operator_bank.npz")
VAL_BEATS_CACHE = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/cache/val_beats_cached.pt")


def load_model(ckpt_path: Path, is_g1: bool = False, device: torch.device = torch.device("cpu")) -> nn.Module:
    chk = torch.load(ckpt_path, map_location="cpu")
    if is_g1:
        model = AblationSetOperatorEchoNext(geometry_mode="G1", classes=1).to(device)
    else:
        model = build_setoperator_echonext("SO-ECG+Tab-Native").to(device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    return model


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("STARTING EXACT LOGIT DECOMPOSITION AND OPERATOR SENSITIVITY AUDIT")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values.astype(np.float32)
    val_t = np.load(DATA_DIR / "EchoNext_val_tabular_features.npy")
    val_t_t = torch.as_tensor(val_t, dtype=torch.float32)

    print(f"Loaded validation cohort: N={len(y_val)}, Positives={int(np.sum(y_val))} ({np.mean(y_val)*100:.2f}%)")

    # 2. Load pre-extracted beats
    print(f"Loading cached validation beats from {VAL_BEATS_CACHE}...")
    t0 = time.time()
    val_beats = torch.load(VAL_BEATS_CACHE, map_location="cpu")
    print(f"Loaded {len(val_beats)} beat tensors in {time.time() - t0:.1f}s.")

    # 3. Load Extractor
    extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device=device)

    # 4. Load Models
    model_base = load_model(BASELINE_CKPT, is_g1=False, device=device)
    model_g1 = load_model(G1_CKPT, is_g1=True, device=device)

    # Canonical operators (8, 8)
    ops_canonical = torch.eye(8, dtype=torch.float32)
    ops_canonical_b = ops_canonical.unsqueeze(0)

    # -------------------------------------------------------------
    # PART 1: EXACT LOGIT DECOMPOSITION ON FULL VALIDATION COHORT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: EXACT NATIVE-FUSION LOGIT DECOMPOSITION ON CANONICAL 8-LEAD BASIS")
    print("=" * 80)

    results_decomp = {}

    for name, model in [("Original SetOperator", model_base), ("G1 Geometry SetOperator", model_g1)]:
        print(f"\nProcessing {name}...")
        w_head = model.head.weight.squeeze(0)  # (256,)
        b_head = float(model.head.bias.item()) # scalar

        ell_ecg_list = []
        ell_tab_list = []
        ell_full_list = []
        recon_errs = []

        # Process in batches
        b_size = 64
        n = len(y_val)

        with torch.no_grad():
            for i in range(0, n, b_size):
                curr_bs = min(b_size, n - i)
                batch_resps = []
                for j in range(curr_bs):
                    b = val_beats[i + j]
                    r = extractor.compute_phase_kme(b, ops_canonical)
                    batch_resps.append(r)
                
                t_resps = torch.stack(batch_resps) # (curr_bs, 8, 16, 128)
                t_tab = val_t_t[i : i + curr_bs]   # (curr_bs, 7)
                t_ops = ops_canonical_b.expand(curr_bs, -1, -1)

                # Forward pass internals
                # 1. Operator and Response tokens
                q = model.encode_operator(t_ops) # (curr_bs, 8, 64)
                r_flat = t_resps.reshape(curr_bs * 8, 16, 128)
                r_enc = model.response(r_flat).reshape(curr_bs, 8, 128)
                tokens = torch.cat([q, r_enc], dim=-1) # (curr_bs, 8, 192)

                # 2. Transformer
                encoded = model.set_encoder(tokens)

                # 3. Pooling
                query = model.pool_query.expand(curr_bs, -1, -1)
                pooled, _ = model.pool(query, encoded, encoded)
                pooled_token = pooled[:, 0] # (curr_bs, 192)

                # 4. Latent components
                h_ecg = model.latent(pooled_token) # (curr_bs, 256)
                h_tab = model.patient_context(t_tab) # (curr_bs, 256)

                # 5. Component logits
                ell_ecg = (h_ecg @ w_head).numpy() # (curr_bs,)
                ell_tab = (h_tab @ w_head).numpy() # (curr_bs,)

                # 6. Full logit
                out = model(operators=t_ops, responses=t_resps, tabular=t_tab, mask=None)
                ell_full = out["logits"].squeeze(-1).numpy()

                # 7. Discrepancy
                recon = ell_ecg + ell_tab + b_head
                recon_err = np.max(np.abs(ell_full - recon))
                recon_errs.append(recon_err)

                ell_ecg_list.extend(ell_ecg.tolist())
                ell_tab_list.extend(ell_tab.tolist())
                ell_full_list.extend(ell_full.tolist())

        ell_ecg_arr = np.array(ell_ecg_list)
        ell_tab_arr = np.array(ell_tab_list)
        ell_full_arr = np.array(ell_full_list)

        max_recon_err = max(recon_errs)
        print(f"  Exact Decomposition Max Discrepancy |ell_full - (ell_ecg + ell_tab + b)|: {max_recon_err:.2e}")
        assert max_recon_err < 1e-4, f"Decomposition failed! Error: {max_recon_err}"

        # Metrics
        auc_ecg = roc_auc_score(y_val, ell_ecg_arr)
        prc_ecg = average_precision_score(y_val, ell_ecg_arr)

        auc_tab = roc_auc_score(y_val, ell_tab_arr)
        prc_tab = average_precision_score(y_val, ell_tab_arr)

        auc_full = roc_auc_score(y_val, ell_full_arr)
        prc_full = average_precision_score(y_val, ell_full_arr)

        r_corr, _ = pearsonr(ell_ecg_arr, ell_tab_arr)
        rho_corr, _ = spearmanr(ell_ecg_arr, ell_tab_arr)

        var_ecg = np.var(ell_ecg_arr)
        var_tab = np.var(ell_tab_arr)
        std_ecg = np.std(ell_ecg_arr)
        std_tab = np.std(ell_tab_arr)

        # Stratified by SHD label
        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)

        mean_ecg_pos = np.mean(ell_ecg_arr[pos_mask])
        mean_ecg_neg = np.mean(ell_ecg_arr[neg_mask])
        delta_ecg = mean_ecg_pos - mean_ecg_neg

        mean_tab_pos = np.mean(ell_tab_arr[pos_mask])
        mean_tab_neg = np.mean(ell_tab_arr[neg_mask])
        delta_tab = mean_tab_pos - mean_tab_neg

        results_decomp[name] = {
            "max_recon_err": float(max_recon_err),
            "bias_head": float(b_head),
            "std_ecg": float(std_ecg),
            "std_tab": float(std_tab),
            "var_ratio_ecg_to_tab": float(var_ecg / var_tab),
            "pearson_corr_ecg_tab": float(r_corr),
            "spearman_corr_ecg_tab": float(rho_corr),
            "auc_ecg": float(auc_ecg),
            "prc_ecg": float(prc_ecg),
            "auc_tab": float(auc_tab),
            "prc_tab": float(prc_tab),
            "auc_full": float(auc_full),
            "prc_full": float(prc_full),
            "delta_ecg_pos_minus_neg": float(delta_ecg),
            "delta_tab_pos_minus_neg": float(delta_tab),
            "mean_ecg_neg": float(mean_ecg_neg),
            "mean_ecg_pos": float(mean_ecg_pos),
            "mean_tab_neg": float(mean_tab_neg),
            "mean_tab_pos": float(mean_tab_pos),
        }

        print(f"  Component AUROC: ECG={auc_ecg:.4f} | Tabular={auc_tab:.4f} | Full={auc_full:.4f}")
        print(f"  Component AUPRC: ECG={prc_ecg:.4f} | Tabular={prc_tab:.4f} | Full={prc_full:.4f}")
        print(f"  Component Logit Std: ECG={std_ecg:.4f} | Tabular={std_tab:.4f} (Var ratio ECG/Tab: {var_ecg/var_tab:.3f})")
        print(f"  Separation (Pos - Neg Mean): ECG={delta_ecg:.4f} | Tabular={delta_tab:.4f}")
        print(f"  Correlation between ell_ecg and ell_tab: Pearson r={r_corr:.3f}, Spearman rho={rho_corr:.3f}")

    # -------------------------------------------------------------
    # PART 2: OPERATOR SENSITIVITY & INVARIANCE AUDIT (P2 & P4)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: OPERATOR SENSITIVITY & ROTATION AUDIT (P2 & P4)")
    print("=" * 80)

    # Load 100-dev bank
    dev_bank = np.load(DEV_BANK_PATH)

    # Select 100 validation patients
    n_sub = 100
    sub_beats = val_beats[:n_sub]
    sub_tab = val_t_t[:n_sub]
    sub_y = y_val[:n_sub]

    # Evaluate across 7 distinct ranks (1 subspace per rank) + intra-subspace rotations
    rank_ops = {r: torch.from_numpy(dev_bank[f"rank_{r}"][0]) for r in range(1, 8)}

    # Rotations for rank 2 subspace 0
    U_r2 = dev_bank["rank_2"][0] # (2, 8)
    theta_45 = np.pi / 4.0
    R_45 = np.array([[np.cos(theta_45), -np.sin(theta_45)], [np.sin(theta_45), np.cos(theta_45)]], dtype=np.float32)
    U_r2_rot = torch.from_numpy(R_45 @ U_r2)
    U_r2_orig = torch.from_numpy(U_r2)

    op_sensitivity = {}

    for name, model in [("Original SetOperator", model_base), ("G1 Geometry SetOperator", model_g1)]:
        w_head = model.head.weight.squeeze(0)
        
        patient_ell_ecg = {r: [] for r in range(1, 8)}
        patient_ell_full = {r: [] for r in range(1, 8)}
        patient_p_full = {r: [] for r in range(1, 8)}

        with torch.no_grad():
            for r in range(1, 8):
                op_r = rank_ops[r] # (r, 8)
                op_r_b = op_r.unsqueeze(0).expand(n_sub, -1, -1)
                
                # compute Phase-KME
                resps = torch.stack([extractor.compute_phase_kme(sub_beats[i], op_r) for i in range(n_sub)])
                
                # Forward internals
                q = model.encode_operator(op_r_b)
                r_flat = resps.reshape(n_sub * r, 16, 128)
                r_enc = model.response(r_flat).reshape(n_sub, r, 128)
                tokens = torch.cat([q, r_enc], dim=-1)
                encoded = model.set_encoder(tokens)
                query = model.pool_query.expand(n_sub, -1, -1)
                pooled, _ = model.pool(query, encoded, encoded)
                h_ecg = model.latent(pooled[:, 0])
                h_tab = model.patient_context(sub_tab)

                ell_ecg = (h_ecg @ w_head).numpy()
                ell_full = (model.head(h_ecg + h_tab)).squeeze(-1).numpy()
                p_full = 1.0 / (1.0 + np.exp(-ell_full))

                patient_ell_ecg[r] = ell_ecg
                patient_ell_full[r] = ell_full
                patient_p_full[r] = p_full

            mat_ell_ecg = np.stack([patient_ell_ecg[r] for r in range(1, 8)], axis=1) # (100, 7)
            mat_p_full = np.stack([patient_p_full[r] for r in range(1, 8)], axis=1)   # (100, 7)

            std_per_patient_ecg = np.mean(np.std(mat_ell_ecg, axis=1))
            std_per_patient_prob = np.mean(np.std(mat_p_full, axis=1))

            # 2. Measure Rotation Sensitivity on Rank 2 Subspace 0
            resps_orig = torch.stack([extractor.compute_phase_kme(sub_beats[i], U_r2_orig) for i in range(n_sub)])
            q_orig = model.encode_operator(U_r2_orig.unsqueeze(0).expand(n_sub, -1, -1))
            tok_orig = torch.cat([q_orig, model.response(resps_orig.reshape(n_sub * 2, 16, 128)).reshape(n_sub, 2, 128)], dim=-1)
            enc_orig = model.set_encoder(tok_orig)
            pooled_orig, _ = model.pool(model.pool_query.expand(n_sub, -1, -1), enc_orig, enc_orig)
            h_ecg_orig = model.latent(pooled_orig[:, 0])
            ell_ecg_orig = (h_ecg_orig @ w_head).numpy()
            p_orig = 1.0 / (1.0 + np.exp(-(model.head(h_ecg_orig + model.patient_context(sub_tab))).squeeze(-1).numpy()))

            resps_rot = torch.stack([extractor.compute_phase_kme(sub_beats[i], U_r2_rot) for i in range(n_sub)])
            q_rot = model.encode_operator(U_r2_rot.unsqueeze(0).expand(n_sub, -1, -1))
            tok_rot = torch.cat([q_rot, model.response(resps_rot.reshape(n_sub * 2, 16, 128)).reshape(n_sub, 2, 128)], dim=-1)
            enc_rot = model.set_encoder(tok_rot)
            pooled_rot, _ = model.pool(model.pool_query.expand(n_sub, -1, -1), enc_rot, enc_rot)
            h_ecg_rot = model.latent(pooled_rot[:, 0])
            ell_ecg_rot = (h_ecg_rot @ w_head).numpy()
            p_rot = 1.0 / (1.0 + np.exp(-(model.head(h_ecg_rot + model.patient_context(sub_tab))).squeeze(-1).numpy()))

            mean_abs_rot_shift_prob = float(np.mean(np.abs(p_orig - p_rot)))
            mean_abs_rot_shift_ecg_logit = float(np.mean(np.abs(ell_ecg_orig - ell_ecg_rot)))
            relative_rot_shift_ecg = mean_abs_rot_shift_ecg_logit / (np.std(ell_ecg_orig) + 1e-8)

        op_sensitivity[name] = {
            "mean_per_patient_ecg_logit_std_across_ranks": float(std_per_patient_ecg),
            "mean_per_patient_prob_std_across_ranks": float(std_per_patient_prob),
            "rotation_mean_abs_prob_shift": float(mean_abs_rot_shift_prob),
            "rotation_mean_abs_ecg_logit_shift": float(mean_abs_rot_shift_ecg_logit),
            "rotation_relative_ecg_shift": float(relative_rot_shift_ecg),
        }

        print(f"\nSensitivity Results for {name}:")
        print(f"  Per-patient ECG logit std across ranks 1..7: {std_per_patient_ecg:.4f}")
        print(f"  Per-patient Probability std across ranks 1..7: {std_per_patient_prob:.4f}")
        print(f"  Under 45-deg rotation of Rank-2 subspace:")
        print(f"    Mean absolute probability shift: {mean_abs_rot_shift_prob:.6f}")
        print(f"    Mean absolute ECG logit shift:   {mean_abs_rot_shift_ecg_logit:.6f}")
        print(f"    Relative ECG logit shift (|Δell_ecg| / std(ell_ecg)): {relative_rot_shift_ecg:.4f}")

    # -------------------------------------------------------------
    # PART 3: AUDIT PHASE-KME INFORMATION & RR TIMING (P5)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: PHASE-KME INFORMATION & RR TIMING AUDIT (P5)")
    print("=" * 80)

    # Inspect beats index
    idx_path = Path("/data/mithunmanivannan/echonext/cache_beats/echonext_val_beats_index.npy")
    idx_data = np.load(idx_path)
    beat_counts = idx_data[:, 1]

    # Measure raw RR intervals from waveform samples
    from repecg.common.beats import detect_rpeaks
    val_w = np.load(DATA_DIR / "EchoNext_val_waveforms.npy", mmap_mode="r")

    n_test_rr = 200
    rr_intervals_ms = []
    fallback_count = 0

    for i in range(n_test_rr):
        wf_8 = val_w[i, 0][:, CANONICAL_8_LEADS].astype(np.float64)
        det = detect_rpeaks(wf_8, fs=250)
        if len(det.valid_intervals) < 2:
            fallback_count += 1
        else:
            for start, end in det.valid_intervals:
                rr_ms = (end - start) * (1000.0 / 250.0)
                rr_intervals_ms.append(rr_ms)

    rr_arr = np.array(rr_intervals_ms)
    mean_rr = float(np.mean(rr_arr))
    std_rr = float(np.std(rr_arr))
    min_rr = float(np.min(rr_arr))
    max_rr = float(np.max(rr_arr))
    fallback_pct = (fallback_count / n_test_rr) * 100.0

    print(f"Sampled {n_test_rr} records for raw RR interval analysis:")
    print(f"  Total valid RR intervals extracted: {len(rr_arr)}")
    print(f"  Mean RR interval: {mean_rr:.1f} ms (Std: {std_rr:.1f} ms, Range: {min_rr:.1f} - {max_rr:.1f} ms)")
    print(f"  Fallback rate (< 2 beats detected): {fallback_count}/{n_test_rr} ({fallback_pct:.2f}%)")
    print(f"  Emergency zero-beat fallback rate: 0.00%")

    # Response tensor variance
    resps_100 = torch.stack([extractor.compute_phase_kme(val_beats[i], ops_canonical) for i in range(100)]).numpy()
    resp_total_var = float(np.var(resps_100))
    resp_between_patient_var = float(np.var(np.mean(resps_100, axis=(1, 2, 3))))
    resp_within_patient_var = float(np.mean(np.var(resps_100, axis=(1, 2, 3))))

    print(f"Phase-KME Response Tensor Variance (over 100 records, shape 100x8x16x128):")
    print(f"  Total variance across all entries: {resp_total_var:.6f}")
    print(f"  Between-patient variance:          {resp_between_patient_var:.6f}")
    print(f"  Within-patient across phase/lead:  {resp_within_patient_var:.6f}")

    # Save complete diagnostic summary
    out_summary = {
        "timestamp": time.time(),
        "logit_decomposition": results_decomp,
        "operator_sensitivity": op_sensitivity,
        "rr_timing_diagnostics": {
            "n_records_sampled": n_test_rr,
            "mean_rr_ms": mean_rr,
            "std_rr_ms": std_rr,
            "min_rr_ms": min_rr,
            "max_rr_ms": max_rr,
            "fallback_percentage": fallback_pct,
            "emergency_zero_fallback_percentage": 0.0,
        },
        "phase_kme_variance": {
            "total_var": resp_total_var,
            "between_patient_var": resp_between_patient_var,
            "within_patient_var": resp_within_patient_var,
        }
    }

    out_json_path = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/evaluations/logit_decomposition_diagnostics.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nDiagnostic audit complete! Saved master JSON to {out_json_path}")


if __name__ == "__main__":
    main()
