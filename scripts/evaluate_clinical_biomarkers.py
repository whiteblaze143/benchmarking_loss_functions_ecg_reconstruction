#!/usr/bin/env python3
"""Evaluate comprehensive clinical biomarkers (QRS, LVH, ST-segment) on PTB-XL."""

import json
import logging
import ast
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import warnings
import concurrent.futures
import statsmodels.api as sm

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from torch.utils.data import Dataset, DataLoader
from scripts.evaluate_comprehensive_registry import load_adapter

class SimpleECGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.pt"), key=lambda p: int(p.stem))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        path = self.files[index]
        signal = torch.load(path, weights_only=True).float()
        return (signal, int(path.stem))

def extract_biomarkers(item_tuple, fs=500):
    """
    Extract QRS duration, LVH voltage criteria, and ST deviation.
    item_tuple is (target_signal, recon_signal) where signal shape is (12, 5000).
    Indices: V1=6, V2=7, V5=10, V6=11.
    """
    target, recon = item_tuple
    
    def _extract_single(sig):
        try:
            # We use Lead V2 (idx 7) for QRS delineation as it generally has clear QRS complexes
            lead_for_del = sig[7]
            signals, info = nk.ecg_process(lead_for_del, sampling_rate=fs)
            rpeaks = info["ECG_R_Peaks"]
            
            if len(rpeaks) < 2:
                return None
                
            _, waves = nk.ecg_delineate(signals["ECG_Clean"].values, rpeaks, sampling_rate=fs, method="dwt")
            
            onsets = np.array(waves["ECG_R_Onsets"])
            roffsets = np.array(waves["ECG_R_Offsets"])
            
            # QRS Duration (ms)
            qrs_durations = []
            valid_qrs_offsets = [] # for J-point
            for p in rpeaks:
                onset_cands = onsets[onsets < p]
                if len(onset_cands) == 0 or np.isnan(onset_cands[-1]):
                    continue
                onset = onset_cands[-1]
                
                roffset_cands = roffsets[roffsets > p]
                if len(roffset_cands) > 0 and not np.isnan(roffset_cands[0]):
                    roffset = roffset_cands[0]
                    if roffset > onset:
                        qrs_durations.append((roffset - onset) / fs * 1000)
                        valid_qrs_offsets.append(roffset)
                        
            if not qrs_durations:
                return None
                
            mean_qrs = np.nanmean(qrs_durations)
            
            # Sokolow-Lyon LVH (S in V1 + R in V5)
            # R peak in V5 (idx 10): 
            v5 = sig[10]
            r_amps_v5 = [v5[p] for p in rpeaks if p < len(v5)]
            mean_r_v5 = np.nanmean(r_amps_v5) if r_amps_v5 else 0
            
            # S trough in V1 (idx 6): search for minimum between R peak and QRS offset
            v1 = sig[6]
            s_amps_v1 = []
            for p, off in zip(rpeaks, roffsets):
                if not np.isnan(off) and off < len(v1) and p < off:
                    s_amps_v1.append(np.min(v1[int(p):int(off)]))
            mean_s_v1 = np.nanmean(s_amps_v1) if s_amps_v1 else 0
            
            # Voltage criteria: S(V1) depth + R(V5) height
            lvh_voltage = abs(mean_s_v1) + max(0, mean_r_v5)
            
            # ST deviation (at J-point + 60ms) on V2
            st_amps = []
            for off in valid_qrs_offsets:
                j60 = int(off + 0.06 * fs)
                if j60 < len(sig[7]):
                    st_amps.append(sig[7][j60])
            mean_st = np.nanmean(st_amps) if st_amps else 0
            
            return {
                "qrs_ms": mean_qrs,
                "lvh_mv": lvh_voltage,
                "st_mv": abs(mean_st)
            }
        except Exception:
            return None

    t_res = _extract_single(target)
    r_res = _extract_single(recon)
    
    if t_res is not None and r_res is not None:
        return t_res, r_res
    return None

def compute_bland_altman_and_regression(y_true, y_pred):
    """Compute Bland-Altman agreement (bias, LoA), R2, Pearson r, and MAE."""
    diff = y_pred - y_true
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff
    
    mae = float(np.mean(np.abs(diff)))
    r_val, p_val = stats.pearsonr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
    
    # Linear regression R2
    slope, intercept, r_value, _, _ = stats.linregress(y_true, y_pred) if len(y_true) > 1 else (0, 0, 0, 0, 0)
    r2 = float(r_value ** 2)
    
    return {
        "mae": mae,
        "pearson_r": float(r_val),
        "r2": r2,
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_low": loa_low,
        "loa_high": loa_high,
    }

def compute_classification_metrics(y_true, y_pred, y_score, n_bootstraps=1000):
    """Compute AUROC, AUPRC, F1, Sensitivity, Specificity, PPV, NPV with 1000-sample bootstrapped CIs."""
    def _metrics(idx):
        t = y_true[idx]
        p = y_pred[idx]
        s = y_score[idx]
        if np.sum(t) == 0 or np.sum(t) == len(t):
            return [np.nan] * 7
        auroc = roc_auc_score(t, s)
        auprc = average_precision_score(t, s)
        f1 = f1_score(t, p)
        tp = np.sum((t == 1) & (p == 1))
        fn = np.sum((t == 1) & (p == 0))
        tn = np.sum((t == 0) & (p == 0))
        fp = np.sum((t == 0) & (p == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        return [auroc, auprc, f1, sens, spec, ppv, npv]
        
    base_metrics = _metrics(np.arange(len(y_true)))
    
    np.random.seed(42)
    boot_res = []
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        res = _metrics(idx)
        if not np.isnan(res[0]):
            boot_res.append(res)
            
    if not boot_res:
        return base_metrics, [(np.nan, np.nan)] * 7
        
    boot_res = np.array(boot_res)
    cis = [(np.percentile(boot_res[:, i], 2.5), np.percentile(boot_res[:, i], 97.5)) for i in range(7)]
    return base_metrics, cis

def fit_logistic_regression(df, target_col, predictor_col):
    """Fit multivariable logistic regression controlling for age & sex and return Adjusted OR & p-val."""
    clean_df = df[[target_col, predictor_col, "age", "sex"]].dropna()
    if len(clean_df) < 10 or clean_df[target_col].nunique() < 2:
        return np.nan, (np.nan, np.nan), np.nan
        
    y = clean_df[target_col].astype(float)
    X = clean_df[[predictor_col, "age", "sex"]].astype(float)
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        odds_ratio = np.exp(model.params[predictor_col])
        conf = model.conf_int()
        ci_low = np.exp(conf.loc[predictor_col, 0])
        ci_high = np.exp(conf.loc[predictor_col, 1])
        pval = model.pvalues[predictor_col]
        return odds_ratio, (ci_low, ci_high), pval
    except Exception:
        return np.nan, (np.nan, np.nan), np.nan

def compute_fisher_exact(y_true, y_pred):
    """Compute Fisher exact test p-value for 2x2 contingency matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    table = [[tp, fp], [fn, tn]]
    odds_ratio, pval = stats.fisher_exact(table)
    return odds_ratio, pval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    
    # Load metadata
    df_ptb = pd.read_csv(project_root / "data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    df_scp = pd.read_csv(project_root / "data/ptb_xl/scp_statements.csv", index_col=0)
    
    def get_superclass_labels(scp_codes_str):
        try:
            scp_codes = ast.literal_eval(scp_codes_str)
            classes = set()
            for code, likelihood in scp_codes.items():
                if likelihood > 0.0 and code in df_scp.index:
                    diagnostic_class = df_scp.loc[code, 'diagnostic_class']
                    if pd.notna(diagnostic_class):
                        classes.add(diagnostic_class)
            return classes
        except:
            return set()
            
    df_ptb["superclasses"] = df_ptb["scp_codes"].apply(get_superclass_labels)
    df_merged = df_ptb
    
    registry_file = project_root / "results/pareto_models_registry.json"
    with open(registry_file) as f:
        registry = json.load(f)
        
    model_ids = [m["id"] for m in registry["models"]]
    logging.info(f"Evaluating {len(model_ids)} models for rigorous Clinical Biomarkers.")
    
    data_dir = project_root / "data/ptb_xl/tensors/test"
    dataset = SimpleECGDataset(data_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    out_dir = project_root / "results/clinical_biomarkers"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "clinical_metrics.csv"
    
    if summary_file.exists():
        summary_file.unlink()
        
    with open(summary_file, 'w') as f:
        f.write("model_id,target,mae,pearson_r,r2,bland_bias,loa_low,loa_high,auroc,auroc_ci_low,auroc_ci_high,auprc,auprc_ci_low,auprc_ci_high,f1,sens,spec,ppv,npv,adj_or,adj_or_ci_low,adj_or_ci_high,pval_logistic,fisher_pval\n")
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
    
    for spec in registry["models"]:
        model_id = spec["id"]
        logging.info(f"Evaluating {model_id}...")
        
        try:
            adapter = load_adapter(spec, device)
        except Exception as e:
            logging.error(f"Failed to load {model_id}: {e}")
            continue
            
        results_df = []
        
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Processing {model_id}"):
                target = batch[0].to(device)
                ecg_ids = batch[1].numpy()
                try:
                    recon = adapter.reconstruct(target)
                except Exception as e:
                    logging.error(f"Failed to reconstruct batch: {e}")
                    continue
                
                target_np = target.cpu().numpy()
                recon_np = recon.cpu().numpy()
                
                tasks = [(target_np[i], recon_np[i]) for i in range(target_np.shape[0])]
                batch_results = list(executor.map(extract_biomarkers, tasks))
                
                for ecg_id, res in zip(ecg_ids, batch_results):
                    if res is not None:
                        t_res, r_res = res
                        if ecg_id in df_merged.index:
                            meta = df_merged.loc[ecg_id]
                            classes = meta["superclasses"]
                            results_df.append({
                                "ecg_id": ecg_id,
                                "age": meta["age"],
                                "sex": meta["sex"],
                                "CD_label": 1 if "CD" in classes else 0,
                                "HYP_label": 1 if "HYP" in classes else 0,
                                "MI_label": 1 if "MI" in classes else 0,
                                "t_qrs": t_res["qrs_ms"],
                                "r_qrs": r_res["qrs_ms"],
                                "t_lvh": t_res["lvh_mv"],
                                "r_lvh": r_res["lvh_mv"],
                                "t_st": t_res["st_mv"],
                                "r_st": r_res["st_mv"],
                            })
                            
        if not results_df:
            continue
            
        df = pd.DataFrame(results_df)
        df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        
        # Analyze QRS
        ba_qrs = compute_bland_altman_and_regression(df["t_qrs"].values, df["r_qrs"].values)
        y_true_qrs = (df["t_qrs"] > 120).astype(int)
        y_pred_qrs = (df["r_qrs"] > 120).astype(int)
        metrics_qrs, cis_qrs = compute_classification_metrics(y_true_qrs.values, y_pred_qrs.values, df["r_qrs"].values)
        df["pred_prolonged_qrs"] = y_pred_qrs
        or_qrs, or_ci_qrs, pval_qrs = fit_logistic_regression(df, "CD_label", "pred_prolonged_qrs")
        _, fisher_p_qrs = compute_fisher_exact(y_true_qrs.values, y_pred_qrs.values)
        
        with open(summary_file, 'a') as f:
            f.write(f"{model_id},QRS,{ba_qrs['mae']:.4f},{ba_qrs['pearson_r']:.4f},{ba_qrs['r2']:.4f},{ba_qrs['bias']:.4f},{ba_qrs['loa_low']:.4f},{ba_qrs['loa_high']:.4f},{metrics_qrs[0]:.4f},{cis_qrs[0][0]:.4f},{cis_qrs[0][1]:.4f},{metrics_qrs[1]:.4f},{cis_qrs[1][0]:.4f},{cis_qrs[1][1]:.4f},{metrics_qrs[2]:.4f},{metrics_qrs[3]:.4f},{metrics_qrs[4]:.4f},{metrics_qrs[5]:.4f},{metrics_qrs[6]:.4f},{or_qrs:.4f},{or_ci_qrs[0]:.4f},{or_ci_qrs[1]:.4f},{pval_qrs:.4f},{fisher_p_qrs:.4f}\n")
            
        # Analyze LVH
        ba_lvh = compute_bland_altman_and_regression(df["t_lvh"].values, df["r_lvh"].values)
        y_true_lvh = (df["t_lvh"] > 3.5).astype(int)
        y_pred_lvh = (df["r_lvh"] > 3.5).astype(int)
        metrics_lvh, cis_lvh = compute_classification_metrics(y_true_lvh.values, y_pred_lvh.values, df["r_lvh"].values)
        df["pred_lvh"] = y_pred_lvh
        or_lvh, or_ci_lvh, pval_lvh = fit_logistic_regression(df, "HYP_label", "pred_lvh")
        _, fisher_p_lvh = compute_fisher_exact(y_true_lvh.values, y_pred_lvh.values)
        
        with open(summary_file, 'a') as f:
            f.write(f"{model_id},LVH,{ba_lvh['mae']:.4f},{ba_lvh['pearson_r']:.4f},{ba_lvh['r2']:.4f},{ba_lvh['bias']:.4f},{ba_lvh['loa_low']:.4f},{ba_lvh['loa_high']:.4f},{metrics_lvh[0]:.4f},{cis_lvh[0][0]:.4f},{cis_lvh[0][1]:.4f},{metrics_lvh[1]:.4f},{cis_lvh[1][0]:.4f},{cis_lvh[1][1]:.4f},{metrics_lvh[2]:.4f},{metrics_lvh[3]:.4f},{metrics_lvh[4]:.4f},{metrics_lvh[5]:.4f},{metrics_lvh[6]:.4f},{or_lvh:.4f},{or_ci_lvh[0]:.4f},{or_ci_lvh[1]:.4f},{pval_lvh:.4f},{fisher_p_lvh:.4f}\n")
            
        # Analyze ST
        ba_st = compute_bland_altman_and_regression(df["t_st"].values, df["r_st"].values)
        y_true_st = (df["t_st"] > 0.1).astype(int)
        y_pred_st = (df["r_st"] > 0.1).astype(int)
        metrics_st, cis_st = compute_classification_metrics(y_true_st.values, y_pred_st.values, df["r_st"].values)
        df["pred_st"] = y_pred_st
        or_st, or_ci_st, pval_st = fit_logistic_regression(df, "MI_label", "pred_st")
        _, fisher_p_st = compute_fisher_exact(y_true_st.values, y_pred_st.values)
        
        with open(summary_file, 'a') as f:
            f.write(f"{model_id},ST,{ba_st['mae']:.4f},{ba_st['pearson_r']:.4f},{ba_st['r2']:.4f},{ba_st['bias']:.4f},{ba_st['loa_low']:.4f},{ba_st['loa_high']:.4f},{metrics_st[0]:.4f},{cis_st[0][0]:.4f},{cis_st[0][1]:.4f},{metrics_st[1]:.4f},{cis_st[1][0]:.4f},{cis_st[1][1]:.4f},{metrics_st[2]:.4f},{metrics_st[3]:.4f},{metrics_st[4]:.4f},{metrics_st[5]:.4f},{metrics_st[6]:.4f},{or_st:.4f},{or_ci_st[0]:.4f},{or_ci_st[1]:.4f},{pval_st:.4f},{fisher_p_st:.4f}\n")

if __name__ == "__main__":
    main()
