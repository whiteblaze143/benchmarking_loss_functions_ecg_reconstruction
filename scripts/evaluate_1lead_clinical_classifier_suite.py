#!/usr/bin/env python3
"""Unified 1-Lead Downstream Clinical Classifier & Biomarker Evaluation Suite.

Evaluates 1-lead ECG reconstruction models across:
1. ECGFounder 150-Task Foundation Model Classifier on PTB-XL (test split, N=2,198)
2. EchoNext 12-Task Structural Heart Disease (SHD) Foundation Model on EchoNext (N=1,000)
3. SemiSeg ViT-Tiny Deep Wave Delineation (replacing NeuroKit):
   - QRS Duration & Conduction Delay (>120 ms)
   - P, QRS, T wave segmentation IoU & mIoU
4. Sokolow-Lyon LVH Index (>3.5 mV)
5. PreSACAN / Representation Variance Retention & Spurious Coupling across V1-V6
6. Multi-lead Signal Fidelity & Bland-Altman Agreement

Supports native GPU acceleration (cuda:0) and strict 1-lead input contracts.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import json
import logging
import math
import os
import sys
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import statsmodels.api as sm

torch.backends.cudnn.enabled = False

_ROOT = Path(__file__).resolve().parents[1]
for dependency in (_ROOT / "external/semiseg/runtime_deps", _ROOT / "external/semiseg/semi-seg-ecg/src", _ROOT):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import models.backbones as vendor_backbones
import models.decode_heads as vendor_heads
from models.encoder_decoder import EncoderDecoder

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from scripts.ecgfounder_classifier import (
    load_ecgfounder,
    load_ptbxl_labels,
    load_task_names,
    preprocess_ecgfounder,
)
from scripts.echonext_classifier import (
    EchoNextMiniModel,
    SHD_TASKS,
    load_echonext_test_metadata,
)
from scripts.evaluate_echonext import load_and_validate as load_echonext_waveforms
from unified_latents.engineering.models.three_d_theta_reconstruction import ThreeDThetaECGAIM
from unified_latents.engineering.experimental.aim_1_lead import (
    build_alitok_vae_1d,
    mask_unobserved_leads,
)
from unified_latents.engineering.utils.regimes import make_lead_indices
from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import build_wavelet_ecg_aim
from scripts.train_3dtheta_ablation import CELL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
OBSERVED_LEAD_IDX = 0  # Fixed Lead I deployment
MISSING_LEAD_INDICES = tuple(range(1, 12))
MISSING_LEAD_NAMES = tuple(LEAD_NAMES[i] for i in MISSING_LEAD_INDICES)
EVALUATION_VERSION = "1lead_clinical_v1"
PATIENT_BOOTSTRAPS = 500


# ==============================================================================
# SemiSeg ViT-Tiny Delineator Loader & Boundary Extraction
# ==============================================================================

def load_semiseg_delineator(device: torch.device):
    ckpt = _ROOT / "results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42/best-MeanIoU.pth"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    backbone_name, backbone_kwargs = next(iter(config["backbone"].items()))
    if backbone_name == "vit_seg_tiny":
        backbone_name = "vit_tiny"
    head_name, head_kwargs = next(iter(config["decode_head"].items()))
    model = EncoderDecoder(
        backbone=getattr(vendor_backbones, backbone_name)(**backbone_kwargs),
        decode_head=getattr(vendor_heads, head_name)(**head_kwargs),
        decode_head_loss=torch.nn.CrossEntropyLoss(),
        use_latent_projection=bool(config.get("use_latent_projection", False)),
        projection_in_dim=config.get("projection_in_dim"),
        projection_out_dim=config.get("projection_out_dim"),
    )
    model.load_state_dict(payload["model_ema"], strict=True)
    return model.float().to(device).eval()


def extract_semiseg_boundaries(mask_500hz: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for class_id, wave in ((1, "P"), (2, "QRS"), (3, "T")):
        active = (mask_500hz == class_id)
        changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1).astype(np.int64)
        stops = (np.flatnonzero(changes == -1) - 1).astype(np.int64)
        output[f"{wave}_onset"] = starts
        output[f"{wave}_offset"] = stops
    return output


def extract_qrs_duration(mask_500hz: np.ndarray) -> float:
    bounds = extract_semiseg_boundaries(mask_500hz)
    onsets, offsets = bounds["QRS_onset"], bounds["QRS_offset"]
    if len(onsets) == 0 or len(offsets) == 0:
        return np.nan
    valid = []
    for on in onsets:
        cand = offsets[offsets > on]
        if len(cand) > 0:
            dur = (cand[0] - on + 1) / 500.0 * 1000.0
            if 40.0 <= dur <= 300.0:
                valid.append(dur)
    return float(np.median(valid)) if valid else np.nan


# ==============================================================================
# Statistical & Agreement Functions
# ==============================================================================

def compute_bland_altman_and_regression(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid_idx = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid_idx], y_pred[valid_idx]
    if len(y_true) < 2:
        return {"mae": np.nan, "pearson_r": np.nan, "r2": np.nan, "bias": np.nan, "sd_diff": np.nan, "loa_low": np.nan, "loa_high": np.nan}
    diff = y_pred - y_true
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff
    mae = float(np.mean(np.abs(diff)))
    r_val, _ = stats.pearsonr(y_true, y_pred)
    _, _, r_value, _, _ = stats.linregress(y_true, y_pred)
    r2 = float(r_value ** 2)
    return {"mae": mae, "pearson_r": float(r_val), "r2": r2, "bias": bias, "sd_diff": sd_diff, "loa_low": loa_low, "loa_high": loa_high}


def compute_classification_metrics(y_true, y_pred, y_score, n_bootstraps=50):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    def _metrics(idx):
        t, p, s = y_true[idx], y_pred[idx], y_score[idx]
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
    if n_bootstraps <= 1:
        return base_metrics, [(np.nan, np.nan)] * 7
    rng = np.random.default_rng(42)
    boot_res = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        res = _metrics(idx)
        if not np.isnan(res[0]):
            boot_res.append(res)
    if not boot_res:
        return base_metrics, [(np.nan, np.nan)] * 7
    boot_res = np.array(boot_res)
    cis = [(float(np.percentile(boot_res[:, i], 2.5)), float(np.percentile(boot_res[:, i], 97.5))) for i in range(7)]
    return base_metrics, cis


def compute_fisher_exact(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    table = [[tp, fp], [fn, tn]]
    try:
        _, pval = stats.fisher_exact(table)
        return float(pval)
    except Exception:
        return np.nan


def fit_logistic_regression(df, target_col, predictor_col):
    if "age" not in df.columns or "sex" not in df.columns:
        return np.nan, (np.nan, np.nan), np.nan
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
        return float(odds_ratio), (float(ci_low), float(ci_high)), float(pval)
    except Exception:
        return np.nan, (np.nan, np.nan), np.nan


def _expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        return np.nan
    value = 0.0
    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]
        included = (probabilities >= lower) & (probabilities < upper) if i < bins - 1 else (probabilities >= lower) & (probabilities <= upper)
        if np.any(included):
            value += np.mean(included) * abs(np.mean(labels[included]) - np.mean(probabilities[included]))
    return float(value)


def _binary_probability_metric(labels, probabilities, metric):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim == 2:
        values = [_binary_probability_metric(labels[:, i], probabilities[:, i], metric) for i in range(labels.shape[1])]
        finite = [v for v in values if np.isfinite(v)]
        return float(np.mean(finite)) if finite else np.nan
    if metric == "auroc":
        return float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    if metric == "auprc":
        return float(average_precision_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    if metric == "brier":
        return float(np.mean((probabilities - labels) ** 2))
    if metric == "ece":
        return _expected_calibration_error(labels, probabilities)
    raise ValueError(f"Unsupported metric: {metric}")


def patient_cluster_bootstrap_delta(labels, reference, reconstructed, patient_ids, metric, n_bootstraps=PATIENT_BOOTSTRAPS, seed=42):
    labels = np.asarray(labels)
    reference = np.asarray(reference, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    patient_ids = np.asarray(patient_ids)
    ref_val = _binary_probability_metric(labels, reference, metric)
    recon_val = _binary_probability_metric(labels, reconstructed, metric)
    point_delta = recon_val - ref_val

    unique_pts = np.unique(patient_ids)
    pt_rows = {pt: np.flatnonzero(patient_ids == pt) for pt in unique_pts}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_pts, size=len(unique_pts), replace=True)
        rows = np.concatenate([pt_rows[pt] for pt in sampled_pts])
        r_boot = _binary_probability_metric(labels[rows], reference[rows], metric)
        rec_boot = _binary_probability_metric(labels[rows], reconstructed[rows], metric)
        d = rec_boot - r_boot
        if np.isfinite(d):
            deltas.append(d)
    if not deltas:
        return {"reference": ref_val, "reconstruction": recon_val, "delta": point_delta, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n_records": len(labels), "n_patients": len(unique_pts), "n_bootstraps": 0}
    deltas = np.asarray(deltas)
    p_pos = (np.sum(deltas <= 0) + 1) / (len(deltas) + 1)
    p_neg = (np.sum(deltas >= 0) + 1) / (len(deltas) + 1)
    return {
        "reference": ref_val,
        "reconstruction": recon_val,
        "delta": point_delta,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float(min(1.0, 2.0 * min(p_pos, p_neg))),
        "n_records": len(labels),
        "n_patients": len(unique_pts),
        "n_bootstraps": len(deltas),
    }


# ==============================================================================
# Model Loading & Reconstruction Wrapper
# ==============================================================================

class Unified1LeadReconstructor:
    """Universal inference wrapper providing bit-exact Lead-I imputation."""

    def __init__(self, model_id: str, ckpt_path: Path, device: torch.device):
        self.model_id = model_id
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.model, self.arch_type = self._load_model()
        self.model = self.model.to(self.device).eval()

    def _load_model(self) -> Tuple[nn.Module, str]:
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.ckpt_path}")
        
        payload = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

        # 1. 3D-Theta Family (D0 - D8)
        base_id = self.model_id.replace("_s42_l0", "").replace("_s200_l0", "").replace("_s1337_l0", "")
        if base_id in CELL_CONFIGS:
            cfg = CELL_CONFIGS[base_id]
            model = ThreeDThetaECGAIM(
                code_mode=cfg["code_mode"],
                fusion=cfg["fusion"],
                width=768,
                encoder_depth=8,
                decoder_depth=4,
                heads=12,
            )
            model.load_state_dict(sd, strict=True)
            return model, "3dtheta"

        # 2. Wavelet MTL Family (conv15e_* and lean2_*)
        if self.model_id.startswith("conv15e_") or self.model_id.startswith("lean2_"):
            cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
            no_del = cfg.get("no_delineation_head", False)
            use_wavelet = cfg.get("use_wavelet_branch", False)
            ssl_mode = cfg.get("ssl_mode", "none")
            
            wavelet_kwargs = {
                "target_len": 5000,
                "patch_size": cfg.get("patch_size", 25),
                "width": cfg.get("width", 768),
                "encoder_depth": cfg.get("encoder_depth", 8),
                "decoder_depth": cfg.get("decoder_depth", 4),
                "heads": cfg.get("heads", 12),
                "lead_conditioning_mode": cfg.get("lead_conditioning_mode", "learned"),
                "use_relative_geometry": cfg.get("use_relative_geometry", False),
                "use_spatial_film": cfg.get("use_spatial_film", False),
                "spatial_gain_init": cfg.get("spatial_gain_init", 0.1),
                "geometry_control": cfg.get("geometry_control", "standard"),
                "use_wavelet_branch": use_wavelet,
                "ssl_mode": ssl_mode,
                "use_delineation_head": not no_del,
                "predict_fiducials": cfg.get("predict_fiducials", False),
                "wavelet_encoder": cfg.get("wavelet_encoder", "timesformer"),
                "wavelet_dim": cfg.get("wavelet_dim", 192),
                "wavelet_depth": cfg.get("wavelet_depth", 2),
                "wavelet_heads": cfg.get("wavelet_heads", 6),
                "wavelet_conv_hidden": cfg.get("wavelet_conv_hidden", 96),
                "wavelet_fusion": cfg.get("wavelet_fusion", "gated_add"),
                "fusion_heads": cfg.get("fusion_heads", 8),
                "view_a": cfg.get("view_a", "magnitude"),
                "view_b": cfg.get("view_b", "phase_sin" if "phase_sin" in str(cfg.get("view_b")) else cfg.get("view_b", "phase")),
                "view_a_bank": cfg.get("view_a_bank", "morlet"),
                "view_b_bank": cfg.get("view_b_bank", "morlet"),
                "view_b_custom_wavelet_asset": cfg.get("view_b_custom_wavelet_asset", None),
                "n_scales": cfg.get("n_scales", 32),
                "min_freq_hz": cfg.get("min_freq_hz", 0.5),
                "max_freq_hz": cfg.get("max_freq_hz", 45.0),
                "morlet_cycles": cfg.get("morlet_cycles", 6.0),
            }
            model = build_wavelet_ecg_aim(**wavelet_kwargs)
            model.load_state_dict(sd, strict=False)
            self.is_zscore = bool(cfg.get("zscore_norm", False))
            return model, "wavelet_mtl"

        # 3. AliTok / Factorial / Spatial Grid (spatial_1lead_*, factorial_ecg_aim_*)
        target_len = payload.get("target_len", 5000) if isinstance(payload, dict) else 5000
        patch_size = payload.get("alitok_patch_size", 25) if isinstance(payload, dict) else 25
        enc_depth = payload.get("alitok_encoder_depth", 8) if isinstance(payload, dict) else 8
        dec_depth = payload.get("alitok_decoder_depth", 4) if isinstance(payload, dict) else 4
        width = payload.get("alitok_width", 768) if isinstance(payload, dict) else 768
        heads = payload.get("alitok_heads", 12) if isinstance(payload, dict) else 12
        lead_cond = payload.get("lead_conditioning_mode", "learned") if isinstance(payload, dict) else "learned"
        use_rel_geom = payload.get("use_relative_geometry", False) if isinstance(payload, dict) else False
        use_spatial_film = payload.get("use_spatial_film", False) if isinstance(payload, dict) else False
        spatial_gain = payload.get("spatial_gain_init", 0.1) if isinstance(payload, dict) else 0.1
        geom_control = payload.get("geometry_control", "standard") if isinstance(payload, dict) else "standard"
        arch = payload.get("architecture", "ecg_aim_v1") if isinstance(payload, dict) else "ecg_aim_v1"
        arch_name = {
            "ecg_aim": "ecg_aim_v1",
            "ecg_aim_spatial": "ecg_aim_spatial_v1",
            "ecg_aim_panorama_author": "ecg_aim_panorama_author_v1",
            "ecg_aim_exact_theta": "ecg_aim_exact_theta_factorial_v1",
        }.get(arch, arch)

        model = build_alitok_vae_1d(
            architecture=arch_name,
            target_len=target_len,
            patch_size=patch_size,
            encoder_depth=enc_depth,
            decoder_depth=dec_depth,
            encoder_width=width,
            decoder_width=width,
            encoder_heads=heads,
            decoder_heads=heads,
            lead_conditioning_mode=lead_cond,
            use_relative_geometry=use_rel_geom,
            use_spatial_film=use_spatial_film,
            spatial_gain_init=spatial_gain,
            geometry_control=geom_control,
        )
        model.load_state_dict(sd, strict=False)
        self.is_zscore = False
        return model, "alitok"

    @torch.inference_mode()
    def reconstruct(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms: [B, 12, 5000] (real physical millivolts).
        
        Returns reconstructed [B, 12, 5000] with exact Lead I passthrough.
        """
        waveforms = waveforms.float().to(self.device)
        B = waveforms.shape[0]

        if getattr(self, "is_zscore", False):
            m = waveforms.mean(dim=(-2, -1), keepdim=True)
            s = waveforms.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            inp_waveforms = (waveforms - m) / s
        else:
            inp_waveforms = waveforms

        if self.arch_type == "3dtheta":
            x_source = inp_waveforms[:, 0:1, :]
            res = self.model(x_source, obs_lead_idx=0)
            recon = res["y_pred"]
        else:
            masked = mask_unobserved_leads(inp_waveforms, [OBSERVED_LEAD_IDX])
            lead_idx = make_lead_indices([OBSERVED_LEAD_IDX], B, self.device)
            if hasattr(self.model, "impute_from_regressor"):
                recon = self.model.impute_from_regressor(masked, lead_indices=lead_idx)["y_pred"]
            else:
                recon = self.model(masked, y_full=inp_waveforms, lead_indices=lead_idx, mode="stage1")["y_pred"]

        if getattr(self, "is_zscore", False):
            recon = recon * s + m

        min_len = min(recon.shape[-1], waveforms.shape[-1])
        recon = recon[..., :min_len]
        recon[:, OBSERVED_LEAD_IDX, :] = waveforms[:, OBSERVED_LEAD_IDX, :min_len]
        return recon


# ==============================================================================
# Simple Dataset Loaders
# ==============================================================================

class SimplePTBXLDataset(Dataset):
    def __init__(self, tensor_dir: Path):
        self.files = sorted(list(tensor_dir.glob("*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sig = torch.load(path, map_location="cpu", weights_only=True).float()
        ecg_id = int(path.stem)
        return sig, ecg_id


# ==============================================================================
# Main Evaluator Logic
# ==============================================================================

def evaluate_model(
    model_id: str,
    ckpt_path: Path,
    device: torch.device,
    db_path: Path,
    batch_size: int = 32,
    smoke: bool = False,
    skip_echonext: bool = False,
):
    logging.info("=" * 70)
    logging.info("Starting Clinical Evaluation for: %s", model_id)
    logging.info("Checkpoint: %s | Device: %s", ckpt_path, device)
    logging.info("=" * 70)

    reconstructor = Unified1LeadReconstructor(model_id, ckpt_path, device)
    semiseg_model = load_semiseg_delineator(device)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=60) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS clinical_metrics (
                dataset TEXT, model_id TEXT, target TEXT,
                mae REAL, pearson_r REAL, r2 REAL, bland_bias REAL, loa_low REAL, loa_high REAL,
                auroc REAL, auroc_ci_low REAL, auroc_ci_high REAL,
                auprc REAL, auprc_ci_low REAL, auprc_ci_high REAL,
                f1 REAL, sens REAL, spec REAL, ppv REAL, npv REAL,
                adj_or REAL, adj_or_ci_low REAL, adj_or_ci_high REAL,
                pval_logistic REAL, fisher_pval REAL, evaluation_version TEXT,
                PRIMARY KEY(dataset, model_id, target, evaluation_version)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS paired_inference (
                dataset TEXT, model_id TEXT, endpoint TEXT, metric TEXT,
                reference_value REAL, reconstruction_value REAL, delta REAL,
                ci_low REAL, ci_high REAL, p_value REAL,
                n_records INTEGER, n_patients INTEGER, n_bootstraps INTEGER,
                cluster_source TEXT, evaluation_version TEXT,
                PRIMARY KEY(dataset, model_id, endpoint, metric, evaluation_version)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS presacan_model_summary (
                model_id TEXT, dataset TEXT, evaluation_version TEXT,
                v3_r_direct_r2 REAL, v3_r_direct_slope REAL, v3_r_var_ret_pct REAL,
                v6_r_direct_r2 REAL, v6_r_direct_slope REAL, v6_r_var_ret_pct REAL,
                interlead_r2_real_I_V3 REAL, interlead_r2_recon_I_V3 REAL,
                interlead_r2_real_I_V6 REAL, interlead_r2_recon_I_V6 REAL,
                spurious_coupling_ratio_v3 REAL,
                avg_precordial_var_ret_pct REAL,
                PRIMARY KEY(model_id, dataset, evaluation_version)
            );
        """)

    def write_metric(ds, mid, target, d):
        with sqlite3.connect(db_path, timeout=60) as con:
            con.execute("""
                INSERT OR REPLACE INTO clinical_metrics (
                    dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
                    auroc, auroc_ci_low, auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high,
                    f1, sens, spec, ppv, npv, adj_or, adj_or_ci_low, adj_or_ci_high, pval_logistic, fisher_pval,
                    evaluation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ds, mid, target, d.get("mae"), d.get("pearson_r"), d.get("r2"), d.get("bland_bias"), d.get("loa_low"), d.get("loa_high"),
                d.get("auroc"), d.get("auroc_ci_low"), d.get("auroc_ci_high"), d.get("auprc"), d.get("auprc_ci_low"), d.get("auprc_ci_high"),
                d.get("f1"), d.get("sens"), d.get("spec"), d.get("ppv"), d.get("npv"), d.get("adj_or"), d.get("adj_or_ci_low"), d.get("adj_or_ci_high"),
                d.get("pval_logistic"), d.get("fisher_pval"), EVALUATION_VERSION
            ))

    def write_paired(ds, mid, endpoint, metric, res, source):
        with sqlite3.connect(db_path, timeout=60) as con:
            con.execute("""
                INSERT OR REPLACE INTO paired_inference (
                    dataset, model_id, endpoint, metric, reference_value, reconstruction_value,
                    delta, ci_low, ci_high, p_value, n_records, n_patients, n_bootstraps, cluster_source,
                    evaluation_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ds, mid, endpoint, metric, res["reference"], res["reconstruction"], res["delta"],
                res["ci_low"], res["ci_high"], res["p_value"], res["n_records"], res["n_patients"],
                res["n_bootstraps"], source, EVALUATION_VERSION
            ))

    # -------------------------------------------------------------------------
    # DATASET 1: PTB-XL
    # -------------------------------------------------------------------------
    logging.info("--> Evaluating PTB-XL (Test Cohort)...")
    ptb_data_dir = _ROOT / "data/ptb_xl/tensors/test"
    df_ptb = pd.read_csv(_ROOT / "data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    
    ecgfounder_repo = _ROOT / "ecg_fm_integration/ecgfounder_repo"
    ecgfounder_ckpt = ecgfounder_repo / "checkpoint/12_lead_ECGFounder.pth"
    ecgfounder_tasks_file = ecgfounder_repo / "tasks.txt"
    ecgfounder_tasks = load_task_names(ecgfounder_tasks_file)
    ecgfounder_model = load_ecgfounder(ecgfounder_repo, ecgfounder_ckpt, device, len(ecgfounder_tasks))
    ecgfounder_labels_df = load_ptbxl_labels(ecgfounder_repo / "csv/ptbxl_label.csv")
    labels_map = dict(zip(ecgfounder_labels_df["filename_hr"], ecgfounder_labels_df["ecgfounder_labels"]))

    ptb_ds = SimplePTBXLDataset(ptb_data_dir)
    ptb_loader = DataLoader(ptb_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_y_true = []
    all_y_recon = []
    ecgfounder_records = []
    patient_ids = []
    age_list, sex_list, hyp_labels = [], [], []

    # SemiSeg Delineation Trackers
    all_t_qrs_dur, all_r_qrs_dur = [], []
    all_p_ious, all_qrs_ious, all_t_ious, all_m_ious = [], [], [], []

    with torch.inference_mode():
        for b_idx, (batch_targets, batch_ids) in enumerate(tqdm(ptb_loader, desc=f"PTB-XL {model_id}")):
            target = batch_targets.to(device)
            recon = reconstructor.reconstruct(target)

            target_np = target.detach().cpu().numpy()
            recon_np = recon.detach().cpu().numpy()
            all_y_true.append(target_np)
            all_y_recon.append(recon_np)

            # ECGFounder Forward Pass on GPU
            t_norm = preprocess_ecgfounder(target)
            r_norm = preprocess_ecgfounder(recon)
            probs_t = torch.sigmoid(ecgfounder_model(t_norm)).detach().cpu().numpy()
            probs_r = torch.sigmoid(ecgfounder_model(r_norm)).detach().cpu().numpy()

            # SemiSeg Deep Delineation on Lead II (idx 1, missing lead)
            t_resamp = F.interpolate(target, size=2500, mode="linear", align_corners=False)
            r_resamp = F.interpolate(recon, size=2500, mode="linear", align_corners=False)
            t_std = (t_resamp - t_resamp.mean(dim=-1, keepdim=True)) / (t_resamp.std(dim=-1, keepdim=True) + 1e-6)
            r_std = (r_resamp - r_resamp.mean(dim=-1, keepdim=True)) / (r_resamp.std(dim=-1, keepdim=True) + 1e-6)

            logits_t_ii = semiseg_model(t_std[:, 1:2, :])["seg_logits"]
            logits_r_ii = semiseg_model(r_std[:, 1:2, :])["seg_logits"]
            preds_t_ii = logits_t_ii.argmax(dim=1).repeat_interleave(2, dim=-1)[:, :5000].cpu().numpy()
            preds_r_ii = logits_r_ii.argmax(dim=1).repeat_interleave(2, dim=-1)[:, :5000].cpu().numpy()

            for i, eid in enumerate(batch_ids.numpy()):
                if eid in df_ptb.index:
                    row = df_ptb.loc[eid]
                    pt_id = row["patient_id"]
                    fn_hr = row.get("filename_hr", "")
                    age_list.append(row.get("age", np.nan))
                    sex_list.append(1 if row.get("sex") == 0 else 0)
                    hyp_labels.append(1 if "HYP" in str(row.get("scp_codes", "")) else 0)
                    patient_ids.append(pt_id)
                    if fn_hr in labels_map:
                        ecgfounder_records.append({
                            "gt_labels": labels_map[fn_hr],
                            "target_probs": probs_t[i],
                            "recon_probs": probs_r[i],
                            "patient_id": pt_id
                        })

                # SemiSeg metrics per record
                mask_t = preds_t_ii[i]
                mask_r = preds_r_ii[i]
                t_dur = extract_qrs_duration(mask_t)
                r_dur = extract_qrs_duration(mask_r)
                all_t_qrs_dur.append(t_dur)
                all_r_qrs_dur.append(r_dur)

                # Segment IoU
                q_inter = np.sum((mask_t == 2) & (mask_r == 2))
                q_union = np.sum((mask_t == 2) | (mask_r == 2))
                q_iou = (q_inter / q_union) if q_union > 0 else 1.0

                p_inter = np.sum((mask_t == 1) & (mask_r == 1))
                p_union = np.sum((mask_t == 1) | (mask_r == 1))
                p_iou = (p_inter / p_union) if p_union > 0 else 1.0

                t_inter = np.sum((mask_t == 3) & (mask_r == 3))
                t_union = np.sum((mask_t == 3) | (mask_r == 3))
                t_iou = (t_inter / t_union) if t_union > 0 else 1.0

                all_qrs_ious.append(q_iou)
                all_p_ious.append(p_iou)
                all_t_ious.append(t_iou)
                all_m_ious.append((q_iou + p_iou + t_iou) / 3.0)

            if smoke and b_idx >= 2:
                break

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_recon_all = np.concatenate(all_y_recon, axis=0)

    # 1. ECGFounder 150-Task Metrics
    if ecgfounder_records:
        logging.info("Computing ECGFounder metrics across %d records...", len(ecgfounder_records))
        y_true_founder = np.array([r["gt_labels"] for r in ecgfounder_records])
        ref_probs = np.array([r["target_probs"] for r in ecgfounder_records])
        recon_probs = np.array([r["recon_probs"] for r in ecgfounder_records])
        pts_founder = np.array([r["patient_id"] for r in ecgfounder_records])

        macro_aurocs, macro_auprcs = [], []
        infarct_aurocs, conduction_aurocs, arrhythmia_aurocs, hypertrophy_aurocs = [], [], [], []

        for t_idx, task_name in enumerate(ecgfounder_tasks):
            yt = (y_true_founder[:, t_idx] > 0.5).astype(int)
            ys = recon_probs[:, t_idx]
            if np.sum(yt) >= 5 and np.sum(yt) <= len(yt) - 5:
                try:
                    auc = roc_auc_score(yt, ys)
                    prc = average_precision_score(yt, ys)
                    macro_aurocs.append(auc)
                    macro_auprcs.append(prc)
                    
                    t_upper = task_name.upper()
                    if "INFARCT" in t_upper: infarct_aurocs.append(auc)
                    if "BLOCK" in t_upper or "AVB" in t_upper or "WPW" in t_upper: conduction_aurocs.append(auc)
                    if "FIBRILLATION" in t_upper or "FLUTTER" in t_upper or "TACHYCARDIA" in t_upper or "PVC" in t_upper or "BRADYCARDIA" in t_upper: arrhythmia_aurocs.append(auc)
                    if "HYPERTROPHY" in t_upper or "ENLARGEMENT" in t_upper: hypertrophy_aurocs.append(auc)

                    if t_upper in ["ATRIAL_FIBRILLATION", "ANTERIOR_INFARCT", "INFERIOR_INFARCT", "LEFT_BUNDLE_BRANCH_BLOCK", "LEFT_VENTRICULAR_HYPERTROPHY"]:
                        m, c = compute_classification_metrics(yt, (ys > 0.5).astype(int), ys, n_bootstraps=50)
                        fish_p = compute_fisher_exact(yt, (ys > 0.5).astype(int))
                        clean_name = t_upper.replace(" ", "_").replace("-", "_")
                        write_metric("ptb_xl", model_id, f"ECGFounder_{clean_name}", {
                            "auroc": m[0], "auroc_ci_low": c[0][0], "auroc_ci_high": c[0][1],
                            "auprc": m[1], "auprc_ci_low": c[1][0], "auprc_ci_high": c[1][1],
                            "f1": m[2], "sens": m[3], "spec": m[4], "ppv": m[5], "npv": m[6],
                            "fisher_pval": fish_p
                        })
                except Exception:
                    pass

        write_metric("ptb_xl", model_id, "ECGFounder_Macro_150", {
            "auroc": float(np.mean(macro_aurocs)) if macro_aurocs else np.nan,
            "auprc": float(np.mean(macro_auprcs)) if macro_auprcs else np.nan,
        })
        if infarct_aurocs:
            write_metric("ptb_xl", model_id, "ECGFounder_Category_Infarct", {"auroc": float(np.mean(infarct_aurocs))})
        if conduction_aurocs:
            write_metric("ptb_xl", model_id, "ECGFounder_Category_Conduction", {"auroc": float(np.mean(conduction_aurocs))})
        if arrhythmia_aurocs:
            write_metric("ptb_xl", model_id, "ECGFounder_Category_Arrhythmia", {"auroc": float(np.mean(arrhythmia_aurocs))})
        if hypertrophy_aurocs:
            write_metric("ptb_xl", model_id, "ECGFounder_Category_Hypertrophy", {"auroc": float(np.mean(hypertrophy_aurocs))})

        for m_name in ("auroc", "auprc", "brier", "ece"):
            paired = patient_cluster_bootstrap_delta(y_true_founder, ref_probs, recon_probs, pts_founder, m_name, n_bootstraps=200 if not smoke else 10)
            write_paired("ptb_xl", model_id, "ECGFounder_Macro", m_name, paired, "ptbxl_database.patient_id")

    # 2. SemiSeg Wave Delineation & QRS Duration Biomarkers
    logging.info("Computing SemiSeg wave delineation & QRS duration biomarkers...")
    ba_qrs = compute_bland_altman_and_regression(all_t_qrs_dur, all_r_qrs_dur)
    write_metric("ptb_xl", model_id, "QRS_Duration_SemiSeg", {
        "mae": ba_qrs["mae"], "pearson_r": ba_qrs["pearson_r"], "r2": ba_qrs["r2"],
        "bland_bias": ba_qrs["bias"], "loa_low": ba_qrs["loa_low"], "loa_high": ba_qrs["loa_high"],
    })

    # Conduction Delay (> 120 ms) classification
    t_qrs_arr = np.asarray(all_t_qrs_dur, dtype=float)
    r_qrs_arr = np.asarray(all_r_qrs_dur, dtype=float)
    valid_qrs = np.isfinite(t_qrs_arr) & np.isfinite(r_qrs_arr)
    if np.sum(valid_qrs) >= 10:
        yt_cd = (t_qrs_arr[valid_qrs] > 120.0).astype(int)
        yr_cd = (r_qrs_arr[valid_qrs] > 120.0).astype(int)
        m_cd, c_cd = compute_classification_metrics(yt_cd, yr_cd, r_qrs_arr[valid_qrs], n_bootstraps=50)
        fish_cd = compute_fisher_exact(yt_cd, yr_cd)
        write_metric("ptb_xl", model_id, "Conduction_Delay_120ms_SemiSeg", {
            "auroc": m_cd[0], "auroc_ci_low": c_cd[0][0], "auroc_ci_high": c_cd[0][1],
            "auprc": m_cd[1], "auprc_ci_low": c_cd[1][0], "auprc_ci_high": c_cd[1][1],
            "f1": m_cd[2], "sens": m_cd[3], "spec": m_cd[4], "ppv": m_cd[5], "npv": m_cd[6],
            "fisher_pval": fish_cd
        })

    # SemiSeg Wave Segmentation IoUs
    write_metric("ptb_xl", model_id, "Delineation_mIoU_SemiSeg", {"r2": float(np.mean(all_m_ious))})
    write_metric("ptb_xl", model_id, "Delineation_QRS_IoU_SemiSeg", {"r2": float(np.mean(all_qrs_ious))})
    write_metric("ptb_xl", model_id, "Delineation_P_IoU_SemiSeg", {"r2": float(np.mean(all_p_ious))})
    write_metric("ptb_xl", model_id, "Delineation_T_IoU_SemiSeg", {"r2": float(np.mean(all_t_ious))})

    # 3. Per-Lead Signal Metrics & PreSACAN
    logging.info("Computing per-lead signal metrics and PreSACAN variance retention...")
    for l_idx, l_name in enumerate(LEAD_NAMES):
        yt_l = y_true_all[:, l_idx, :]
        yr_l = y_recon_all[:, l_idx, :]
        r_list, mae_list, mse_list = [], [], []
        for i in range(len(yt_l)):
            yt_c = yt_l[i] - np.mean(yt_l[i])
            yr_c = yr_l[i] - np.mean(yr_l[i])
            denom = np.sqrt(np.sum(yt_c ** 2) * np.sum(yr_c ** 2)) + 1e-8
            r_list.append(np.sum(yt_c * yr_c) / denom)
            mae_list.append(np.mean(np.abs(yt_l[i] - yr_l[i])))
            mse_list.append(np.mean((yt_l[i] - yr_l[i]) ** 2))
        ba = compute_bland_altman_and_regression(yt_l.flatten(), yr_l.flatten())
        write_metric("ptb_xl", model_id, f"Signal_Lead_{l_name}", {
            "mae": float(np.mean(mae_list)),
            "pearson_r": float(np.mean(r_list)),
            "r2": ba["r2"],
            "bland_bias": ba["bias"],
            "loa_low": ba["loa_low"],
            "loa_high": ba["loa_high"],
        })

    # PreSACAN: Nature Bland-Altman regression-to-the-mean & variance retention across V1-V6
    var_rets = []
    presacan_summary = {"model_id": model_id, "dataset": "ptb_xl", "evaluation_version": EVALUATION_VERSION}
    for prec_lead, l_idx in [("V3", 8), ("V6", 11)]:
        yt_lead = y_true_all[:, l_idx, :]
        yr_lead = y_recon_all[:, l_idx, :]
        yt_r = np.max(yt_lead[:, 1000:4000], axis=1)
        yr_r = np.max(yr_lead[:, 1000:4000], axis=1)
        
        var_real = float(np.var(yt_r))
        var_rec = float(np.var(yr_r))
        ret_pct = (var_rec / var_real * 100.0) if var_real > 0 else 0.0
        var_rets.append(ret_pct)

        # Direct agreement regression (yr vs yt)
        slope_d, _, r_d, _, _ = stats.linregress(yt_r, yr_r)

        # PreSACAN error regression (yr - yt vs yt)
        err_r = yr_r - yt_r
        slope_ps, _, r_ps, _, _ = stats.linregress(yt_r, err_r)
        
        y_lead_I = np.max(y_true_all[:, 0, 1000:4000], axis=1)
        _, _, r_inter_real, _, _ = stats.linregress(y_lead_I, yt_r)
        _, _, r_inter_recon, _, _ = stats.linregress(y_lead_I, yr_r)

        if prec_lead == "V3":
            presacan_summary["v3_r_presacan_slope"] = float(slope_ps)
            presacan_summary["v3_r_presacan_r2"] = float(r_ps ** 2)
            presacan_summary["v3_r_direct_r2"] = float(r_d ** 2)
            presacan_summary["v3_r_direct_slope"] = float(slope_d)
            presacan_summary["v3_r_var_ret_pct"] = float(ret_pct)
            presacan_summary["interlead_r2_real_I_V3"] = float(r_inter_real ** 2)
            presacan_summary["interlead_r2_recon_I_V3"] = float(r_inter_recon ** 2)
            presacan_summary["spurious_coupling_ratio_v3"] = float((r_inter_recon ** 2) / (r_inter_real ** 2 + 1e-6))
        else:
            presacan_summary["v6_r_presacan_slope"] = float(slope_ps)
            presacan_summary["v6_r_presacan_r2"] = float(r_ps ** 2)
            presacan_summary["v6_r_direct_r2"] = float(r_d ** 2)
            presacan_summary["v6_r_direct_slope"] = float(slope_d)
            presacan_summary["v6_r_var_ret_pct"] = float(ret_pct)
            presacan_summary["interlead_r2_real_I_V6"] = float(r_inter_real ** 2)
            presacan_summary["interlead_r2_recon_I_V6"] = float(r_inter_recon ** 2)

    # PreSACAN T-Wave in V3
    yt_t_v3 = np.max(y_true_all[:, 8, 3000:4500], axis=1)
    yr_t_v3 = np.max(y_recon_all[:, 8, 3000:4500], axis=1)
    var_real_t = float(np.var(yt_t_v3))
    var_rec_t = float(np.var(yr_t_v3))
    t_ret_pct = (var_rec_t / var_real_t * 100.0) if var_real_t > 0 else 0.0
    slope_t_ps, _, r_t_ps, _, _ = stats.linregress(yt_t_v3, yr_t_v3 - yt_t_v3)
    presacan_summary["v3_t_var_ret_pct"] = float(t_ret_pct)
    presacan_summary["v3_t_presacan_slope"] = float(slope_t_ps)
    presacan_summary["v3_t_presacan_r2"] = float(r_t_ps ** 2)

    presacan_summary["avg_precordial_var_ret_pct"] = float(np.mean(var_rets))
    with sqlite3.connect(db_path, timeout=60) as con:
        con.execute("""
            INSERT OR REPLACE INTO presacan_model_summary (
                model_id, dataset, evaluation_version,
                v3_r_presacan_r2, v3_r_presacan_slope, v3_r_var_ret_pct,
                v3_r_direct_r2, v3_r_direct_slope,
                v6_r_presacan_r2, v6_r_presacan_slope, v6_r_var_ret_pct,
                v6_r_direct_r2, v6_r_direct_slope,
                v3_t_presacan_r2, v3_t_presacan_slope, v3_t_var_ret_pct,
                interlead_r2_real_I_V3, interlead_r2_recon_I_V3,
                interlead_r2_real_I_V6, interlead_r2_recon_I_V6,
                spurious_coupling_ratio_v3,
                avg_precordial_var_ret_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            presacan_summary["model_id"], presacan_summary["dataset"], presacan_summary["evaluation_version"],
            presacan_summary.get("v3_r_presacan_r2"), presacan_summary.get("v3_r_presacan_slope"), presacan_summary.get("v3_r_var_ret_pct"),
            presacan_summary.get("v3_r_direct_r2"), presacan_summary.get("v3_r_direct_slope"),
            presacan_summary.get("v6_r_presacan_r2"), presacan_summary.get("v6_r_presacan_slope"), presacan_summary.get("v6_r_var_ret_pct"),
            presacan_summary.get("v6_r_direct_r2"), presacan_summary.get("v6_r_direct_slope"),
            presacan_summary.get("v3_t_presacan_r2"), presacan_summary.get("v3_t_presacan_slope"), presacan_summary.get("v3_t_var_ret_pct"),
            presacan_summary.get("interlead_r2_real_I_V3"), presacan_summary.get("interlead_r2_recon_I_V3"),
            presacan_summary.get("interlead_r2_real_I_V6"), presacan_summary.get("interlead_r2_recon_I_V6"),
            presacan_summary.get("spurious_coupling_ratio_v3"),
            presacan_summary.get("avg_precordial_var_ret_pct")
        ))

    # 4. Sokolow-Lyon LVH Index
    yt_s_v1 = np.abs(np.min(y_true_all[:, 6, 1000:4000], axis=1))
    yt_r_v5 = np.max(y_true_all[:, 10, 1000:4000], axis=1)
    yr_s_v1 = np.abs(np.min(y_recon_all[:, 6, 1000:4000], axis=1))
    yr_r_v5 = np.max(y_recon_all[:, 10, 1000:4000], axis=1)
    
    t_lvh_val = yt_s_v1 + yt_r_v5
    r_lvh_val = yr_s_v1 + yr_r_v5
    ba_lvh = compute_bland_altman_and_regression(t_lvh_val, r_lvh_val)
    yt_lvh_bin = (t_lvh_val > 3.5).astype(int)
    yr_lvh_bin = (r_lvh_val > 3.5).astype(int)
    m_lvh, c_lvh = compute_classification_metrics(yt_lvh_bin, yr_lvh_bin, r_lvh_val, n_bootstraps=50)
    write_metric("ptb_xl", model_id, "LVH_SokolowLyon", {
        "mae": ba_lvh["mae"], "pearson_r": ba_lvh["pearson_r"], "r2": ba_lvh["r2"],
        "bland_bias": ba_lvh["bias"], "loa_low": ba_lvh["loa_low"], "loa_high": ba_lvh["loa_high"],
        "auroc": m_lvh[0], "auroc_ci_low": c_lvh[0][0], "auroc_ci_high": c_lvh[0][1],
        "auprc": m_lvh[1], "auprc_ci_low": c_lvh[1][0], "auprc_ci_high": c_lvh[1][1],
        "f1": m_lvh[2], "sens": m_lvh[3], "spec": m_lvh[4], "ppv": m_lvh[5], "npv": m_lvh[6]
    })
    # Free PTB-XL tensors and models before EchoNext to prevent RAM accumulation on CPU
    try:
        del y_true_all, y_recon_all, fm_classifier, delineation_model
    except Exception:
        pass
    import gc
    gc.collect()

    # -------------------------------------------------------------------------
    # DATASET 2: EchoNext (Structural Heart Disease)
    # -------------------------------------------------------------------------
    echo_file = _ROOT / "data/echonext/EchoNext_test_waveforms.npy"
    if not skip_echonext and echo_file.exists():
        logging.info("--> Evaluating EchoNext (N=1,000)...")
        echo_data, _ = load_echonext_waveforms(_ROOT / "data/echonext")
        n_echo = min(1000, len(echo_data)) if not smoke else min(64, len(echo_data))
        echonext_model_root = _ROOT / "ecg_fm_integration/echonext_minimodel_repo/7-EchoNext Minimodel"
        shd_classifier = EchoNextMiniModel(echonext_model_root, device)
        shd_metadata, shd_tabular, shd_labels = load_echonext_test_metadata(
            _ROOT / "data/echonext/echonext_metadata_100k.csv",
            shd_classifier.transformer_path,
        )

        ref_shd_chunks = []
        rec_shd_chunks = []
        bs = batch_size
        with torch.inference_mode():
            for start_idx in range(0, n_echo, bs):
                stop_idx = min(start_idx + bs, n_echo)
                batch_np = echo_data.batch(start_idx, stop_idx)
                target_t = torch.from_numpy(batch_np).float().to(device)
                recon_t = reconstructor.reconstruct(target_t)

                ref_shd_chunks.append(shd_classifier.predict_official_waveforms(
                    echo_data.official_normalized_batch(start_idx, stop_idx),
                    shd_tabular[start_idx:stop_idx]
                ))
                rec_shd_chunks.append(shd_classifier.predict_reconstruction_500hz(
                    recon_t,
                    shd_tabular[start_idx:stop_idx]
                ))

        ref_shd = np.concatenate(ref_shd_chunks, axis=0)
        rec_shd = np.concatenate(rec_shd_chunks, axis=0)
        sub_labels = shd_labels[:n_echo]
        sub_meta = shd_metadata.iloc[:n_echo]

        shd_aurocs, shd_auprcs = [], []
        for t_idx, t_name in enumerate(SHD_TASKS):
            yt = sub_labels[:, t_idx]
            ys = rec_shd[:, t_idx]
            if np.sum(yt) >= 3 and np.sum(yt) <= len(yt) - 3:
                try:
                    auc = roc_auc_score(yt, ys)
                    prc = average_precision_score(yt, ys)
                    shd_aurocs.append(auc)
                    shd_auprcs.append(prc)
                except Exception:
                    pass

        mean_shd_auroc = float(np.mean(shd_aurocs)) if shd_aurocs else np.nan
        mean_shd_auprc = float(np.mean(shd_auprcs)) if shd_auprcs else np.nan
        write_metric("echonext", model_id, "EchoNextSHD_Macro_12", {"auroc": mean_shd_auroc, "auprc": mean_shd_auprc})

        for m_name in ("auroc", "auprc", "brier", "ece"):
            paired = patient_cluster_bootstrap_delta(sub_labels, ref_shd, rec_shd, sub_meta["patient_key"].to_numpy(), m_name, n_bootstraps=200 if not smoke else 10)
            write_paired("echonext", model_id, "EchoNextSHD_Macro_12", m_name, paired, "echonext_metadata.patient_key")

    logging.info("✓ Evaluation Complete for %s!", model_id)


def main():
    parser = argparse.ArgumentParser(description="Unified 1-Lead Clinical Classifier Evaluation")
    parser.add_argument("--model-id", type=str, required=True, help="Model identifier")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--db-path", type=str, default="results/clinical_biomarkers_multids/clinical_metrics.db")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test on 3 batches")
    parser.add_argument("--skip-echonext", action="store_true", help="Skip EchoNext evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)
    db_path = _ROOT / args.db_path
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path

    evaluate_model(
        model_id=args.model_id,
        ckpt_path=ckpt_path,
        device=device,
        db_path=db_path,
        batch_size=args.batch_size,
        smoke=args.smoke,
        skip_echonext=args.skip_echonext,
    )


if __name__ == "__main__":
    main()
