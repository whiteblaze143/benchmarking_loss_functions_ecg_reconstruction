#
import sqlite3
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from tqdm import tqdm

torch.set_num_threads(2)

_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluate_clinical_biomarkers_multids import (
    load_adapter, SimplePTBDataset, LEAD_NAMES
)

db_path = str(_ROOT / "results/clinical_biomarkers_multids/clinical_metrics.db")
ptb_data_dir = str(_ROOT / "data/ptb_xl/tensors/test")
registry_path = _ROOT / "results/clinical_biomarkers_model_registry.json"

with open(registry_path) as f:
    reg = json.load(f)

model_specs_by_id = {m["id"]: m for m in reg["models"]}
all_models = [m["id"] for m in reg["models"]]

conn = sqlite3.connect(db_path, timeout=60)
cursor = conn.cursor()
cursor.execute("SELECT model_id FROM presacan_model_summary WHERE dataset='ptb_xl' AND evaluation_version='missing_leads_v2'")
completed_models = set(row[0] for row in cursor.fetchall())
conn.close()

models_to_eval = [m for m in all_models if m not in completed_models]

print(f"=== Presacan Nature Benchmark Engine (Clinical Metrics DB Sync) ===")
print(f"Total registered clinical models: {len(all_models)} (160 U-Net + 62 MS-VAE)")
print(f"Already completed: {len(completed_models)} | Remaining to evaluate: {len(models_to_eval)}")

# 1. Preload all test tensors into memory for maximum speed
print("Preloading test tensors into memory...")
test_dataset = SimplePTBDataset(ptb_data_dir)
all_test_tensors = torch.stack([s[0] for s in test_dataset], dim=0) # shape (N, 12, 5000)
print(f"Loaded test tensor matrix: {all_test_tensors.shape}")

def extract_amplitudes_matrix(sig_Nx12x5000):
    # vectorized across all N records
    N = sig_Nx12x5000.shape[0]
    res_dict = {}
    for idx, lead_name in enumerate(LEAD_NAMES):
        lead_sigs = sig_Nx12x5000[:, idx, :] # (N, 5000)
        r_amps = np.percentile(lead_sigs, 99.0, axis=1)
        s_amps = np.percentile(lead_sigs, 1.0, axis=1)
        st_amps = np.mean(lead_sigs[:, 200:350], axis=1)
        t_amps = np.percentile(lead_sigs, 90.0, axis=1)
        res_dict[lead_name] = {
            "R": r_amps,
            "S": s_amps,
            "ST": st_amps,
            "T": t_amps
        }
    return res_dict

# Extract Real Ground Truth Amplitudes
print("Extracting Real Ground Truth ECG Amplitudes across test set...")
real_dict = extract_amplitudes_matrix(all_test_tensors.numpy())

# Compute Ground Truth Interlead R2
_, _, r_real_I_V3_R, _, _ = stats.linregress(real_dict["I"]["R"], real_dict["V3"]["R"])
real_I_V3_R2 = float(r_real_I_V3_R ** 2)

_, _, r_real_I_V6_R, _, _ = stats.linregress(real_dict["I"]["R"], real_dict["V6"]["R"])
real_I_V6_R2 = float(r_real_I_V6_R ** 2)

_, _, r_real_I_V3_T, _, _ = stats.linregress(real_dict["I"]["T"], real_dict["V3"]["T"])
real_I_V3_T_R2 = float(r_real_I_V3_T ** 2)

print(f"Ground Truth Baseline: Real R^2(Lead I, Lead V3) = {real_I_V3_R2:.4f}")

def evaluate_single_model(mid):
    try:
        device = torch.device("cpu")
        if "msvae" in mid:
            kind = "msvae"
        elif "ecg_aim" in mid or "aim" in mid:
            kind = "ecg_aim"
        else:
            kind = "unet"

        mspec = model_specs_by_id.get(mid, {
            "id": mid,
            "kind": kind,
            "checkpoint": f"checkpoints/{mid}.pt",
            "observed_leads": [0, 1, 7]
        })
        try:
            adapter = load_adapter(mspec, device)
        except Exception as load_err:
            if mid.startswith("f_"):
                alt_id = "factorial_" + mid[2:]
                alt_spec = model_specs_by_id.get(alt_id, {"id": alt_id, "kind": kind, "checkpoint": f"checkpoints/{alt_id}.pt", "observed_leads": [0, 1, 7]})
                adapter = load_adapter(alt_spec, device)
            elif mid.startswith("factorial_") and "msvae" not in mid and "aim" not in mid:
                alt_id = "f_" + mid[len("factorial_"):]
                alt_spec = model_specs_by_id.get(alt_id, {"id": alt_id, "kind": kind, "checkpoint": f"checkpoints/{alt_id}.pt", "observed_leads": [0, 1, 7]})
                adapter = load_adapter(alt_spec, device)
            else:
                raise load_err

        # Reconstruct in clean micro-batches
        recons = []
        batch_size = 64
        with torch.inference_mode():
            for i in range(0, all_test_tensors.shape[0], batch_size):
                b_target = all_test_tensors[i:i+batch_size].to(device)
                b_recon = adapter.reconstruct(b_target).cpu()
                recons.append(b_recon)
        full_recon = torch.cat(recons, dim=0).numpy()

        recon_dict = extract_amplitudes_matrix(full_recon)

        # Statistical Calculations
        detailed_rows = []
        precordial_var_retentions = []

        for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            for feat in ["R", "S", "ST", "T"]:
                y_real = real_dict[lead][feat]
                y_recon = recon_dict[lead][feat]

                error = y_recon - y_real

                real_mean = float(np.mean(y_real))
                real_sd = float(np.std(y_real, ddof=1))
                recon_mean = float(np.mean(y_recon))
                recon_sd = float(np.std(y_recon, ddof=1))

                # Student's t-test
                try:
                    _, p_mean = stats.ttest_rel(y_real, y_recon)
                    p_mean = float(p_mean)
                except:
                    p_mean = np.nan

                # F-test for equality of variances
                var_real = float(np.var(y_real, ddof=1))
                var_recon = float(np.var(y_recon, ddof=1))
                if var_recon > 0 and var_real > 0:
                    f_stat = var_recon / var_real
                    df1 = len(y_recon) - 1
                    df2 = len(y_real) - 1
                    p_var = float(2.0 * min(stats.f.cdf(f_stat, df1, df2), 1.0 - stats.f.cdf(f_stat, df1, df2)))
                else:
                    p_var = np.nan

                # Variance Retention Ratio (%)
                var_ret_pct = float((var_recon / (var_real + 1e-12)) * 100.0) if var_real > 0 else np.nan
                if lead in ["V1", "V3", "V4", "V5", "V6"]:
                    precordial_var_retentions.append(var_ret_pct)

                # Bland-Altman Limits of Agreement
                bland_bias = float(np.mean(error))
                bland_sd = float(np.std(error, ddof=1))
                loa_low = float(bland_bias - 1.96 * bland_sd)
                loa_high = float(bland_bias + 1.96 * bland_sd)

                # Error percentiles
                p5 = float(np.percentile(error, 5.0))
                p10 = float(np.percentile(error, 10.0))
                p90 = float(np.percentile(error, 90.0))
                p95 = float(np.percentile(error, 95.0))

                # Presacan Linear Regression (Error vs Real)
                try:
                    slope, intercept, r_val, _, _ = stats.linregress(y_real, error)
                    presacan_r2 = float(r_val ** 2)
                    presacan_slope = float(slope)
                    presacan_intercept = float(intercept)
                except:
                    presacan_r2, presacan_slope, presacan_intercept = np.nan, np.nan, np.nan

                # Direct Agreement Regression (Recon vs Real)
                try:
                    d_slope, d_intercept, d_r_val, _, _ = stats.linregress(y_real, y_recon)
                    direct_r2 = float(d_r_val ** 2)
                    direct_slope = float(d_slope)
                except:
                    direct_r2, direct_slope = np.nan, np.nan

                detailed_rows.append((
                    "ptb_xl", mid, lead, feat,
                    real_mean, real_sd, recon_mean, recon_sd,
                    p_mean, p_var, var_ret_pct,
                    bland_bias, loa_low, loa_high,
                    p5, p10, p90, p95,
                    presacan_r2, presacan_slope, presacan_intercept,
                    direct_r2, direct_slope,
                    "missing_leads_v2"
                ))

        # Model Summary Metrics
        v3_r_real = real_dict["V3"]["R"]
        v3_r_recon = recon_dict["V3"]["R"]
        v3_r_err = v3_r_recon - v3_r_real
        slope_v3, _, r_v3, _, _ = stats.linregress(v3_r_real, v3_r_err)
        v3_r_presacan_slope = float(slope_v3)
        v3_r_presacan_r2 = float(r_v3 ** 2)
        v3_r_var_ret = float((np.var(v3_r_recon, ddof=1) / (np.var(v3_r_real, ddof=1) + 1e-12)) * 100.0)
        d_slope_v3, _, d_r_v3, _, _ = stats.linregress(v3_r_real, v3_r_recon)
        v3_r_direct_slope = float(d_slope_v3)
        v3_r_direct_r2 = float(d_r_v3 ** 2)

        v6_r_real = real_dict["V6"]["R"]
        v6_r_recon = recon_dict["V6"]["R"]
        v6_r_err = v6_r_recon - v6_r_real
        slope_v6, _, r_v6, _, _ = stats.linregress(v6_r_real, v6_r_err)
        v6_r_presacan_slope = float(slope_v6)
        v6_r_presacan_r2 = float(r_v6 ** 2)
        v6_r_var_ret = float((np.var(v6_r_recon, ddof=1) / (np.var(v6_r_real, ddof=1) + 1e-12)) * 100.0)
        d_slope_v6, _, d_r_v6, _, _ = stats.linregress(v6_r_real, v6_r_recon)
        v6_r_direct_slope = float(d_slope_v6)
        v6_r_direct_r2 = float(d_r_v6 ** 2)

        v3_t_real = real_dict["V3"]["T"]
        v3_t_recon = recon_dict["V3"]["T"]
        v3_t_err = v3_t_recon - v3_t_real
        slope_t_v3, _, r_t_v3, _, _ = stats.linregress(v3_t_real, v3_t_err)
        v3_t_presacan_slope = float(slope_t_v3)
        v3_t_presacan_r2 = float(r_t_v3 ** 2)
        v3_t_var_ret = float((np.var(v3_t_recon, ddof=1) / (np.var(v3_t_real, ddof=1) + 1e-12)) * 100.0)

        # Interlead Correlations
        _, _, r_recon_I_V3, _, _ = stats.linregress(recon_dict["I"]["R"], recon_dict["V3"]["R"])
        recon_I_V3_R2 = float(r_recon_I_V3 ** 2)

        _, _, r_recon_I_V6, _, _ = stats.linregress(recon_dict["I"]["R"], recon_dict["V6"]["R"])
        recon_I_V6_R2 = float(r_recon_I_V6 ** 2)

        _, _, r_recon_I_V3_T, _, _ = stats.linregress(recon_dict["I"]["T"], recon_dict["V3"]["T"])
        recon_I_V3_T_R2 = float(r_recon_I_V3_T ** 2)

        spurious_coupling_ratio = float(recon_I_V3_R2 / (real_I_V3_R2 + 1e-8))
        avg_precordial_var_ret = float(np.mean(precordial_var_retentions))

        summary_row = (
            mid, "ptb_xl", "missing_leads_v2",
            v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret, v3_r_direct_r2, v3_r_direct_slope,
            v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret, v6_r_direct_r2, v6_r_direct_slope,
            v3_t_presacan_r2, v3_t_presacan_slope, v3_t_var_ret,
            real_I_V3_R2, recon_I_V3_R2,
            real_I_V6_R2, recon_I_V6_R2,
            real_I_V3_T_R2, recon_I_V3_T_R2,
            spurious_coupling_ratio, avg_precordial_var_ret
        )

        with sqlite3.connect(db_path, timeout=60) as db_conn:
            db_conn.executemany("""
                INSERT OR REPLACE INTO presacan_clinical_metrics (
                    dataset, model_id, lead, feature,
                    real_mean, real_sd, recon_mean, recon_sd,
                    p_mean, p_var, var_ret_pct,
                    bland_bias, loa_low, loa_high,
                    p5_error, p10_error, p90_error, p95_error,
                    presacan_r2, presacan_slope, presacan_intercept,
                    direct_r2, direct_slope,
                    evaluation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, detailed_rows)

            db_conn.execute("""
                INSERT OR REPLACE INTO presacan_model_summary (
                    model_id, dataset, evaluation_version,
                    v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct, v3_r_direct_r2, v3_r_direct_slope,
                    v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct, v6_r_direct_r2, v6_r_direct_slope,
                    v3_t_presacan_r2, v3_t_presacan_slope, v3_t_var_ret_pct,
                    interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
                    interlead_r2_real_I_V6, interlead_r2_recon_I_V6,
                    interlead_t_r2_real_I_V3, interlead_t_r2_recon_I_V3,
              