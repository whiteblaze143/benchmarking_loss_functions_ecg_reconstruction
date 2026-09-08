#!/usr/bin/env python3
"""Extract 100% Empirical Patient-Level Clinical Predictions and Reconstructions.

Evaluates the FULL 55 benchmark models across:
1. PTB-XL Test Cohort (N = 2,198 ECGs, 1,904 unique patients, fold 10)
2. EchoNext Test Cohort (N = 1,000 paired patients)

Roster (55 Models):
- Track 1: 12 15-Epoch Convergence Models (Wavelet / MTL / SSL)
- Track 2: 9 3D-Theta Geometric Conditioning Models (Ansari D-Series)
- Track 3: 4 Factorial Mask Models (Factorial Baselines)
- Track 4: 30 3-Epoch Spatial Architecture Grid Models (Spatial / Geometry)

For EVERY single patient and EVERY model:
- Reconstructs Lead I -> 12-lead on CUDA:0
- Evaluates SemiSeg ViT-Tiny for QRS duration (ms)
- Evaluates Sokolow-Lyon LVH voltage (mV)
- Evaluates ECGFounder Foundation Model probabilities (Arrhythmia, Conduction, Infarct, Hypertrophy)
- Evaluates EchoNext Foundation Model probabilities (LVEF <= 45%, LVWT >= 13mm, SHD, PASP >= 45mmHg)
- Records continuous biomarker errors, per-lead correlations, and binary diagnostic concordances.

Persists strictly empirical observation records to SQLite:
results/clinical_biomarkers_multids/patient_level_clinical_observations.sqlite
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluate_1lead_clinical_classifier_suite import (
    LEAD_NAMES,
    OBSERVED_LEAD_IDX,
    Unified1LeadReconstructor,
    extract_qrs_duration,
    load_ecgfounder,
    load_ptbxl_labels,
    load_semiseg_delineator,
    load_task_names,
    preprocess_ecgfounder,
)
from scripts.echonext_classifier import (
    EchoNextMiniModel,
    SHD_TASKS,
    load_echonext_test_metadata,
)
from scripts.evaluate_echonext import load_and_validate as load_echonext_waveforms
from scripts.run_1lead_clinical_eval_queue import build_55_model_roster

torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

OUT_DIR = _ROOT / "results/clinical_biomarkers_multids"
DB_PATH = OUT_DIR / "patient_level_clinical_observations.sqlite"


def map_track_to_family(track: str) -> str:
    if "Convergence" in track:
        return "Wavelet / MTL / SSL"
    elif "3D-Theta" in track:
        return "Ansari 3DRECON-QT (D-Series)"
    elif "Factorial" in track:
        return "Factorial Baseline"
    elif "Spatial" in track:
        return "Spatial / Geometry"
    return "Baseline / Other"


def load_ptbxl_test_cohort() -> Tuple[pd.DataFrame, torch.Tensor, List[int]]:
    logging.info("Loading PTB-XL test cohort...")
    csv_path = _ROOT / "data/ptb_xl/ptbxl_database.csv"
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df_test = df[df["strat_fold"] == 10].copy()

    # Demographics & Biometrics
    df_test["age_clean"] = pd.to_numeric(df_test["age"], errors="coerce").fillna(60.0)
    df_test["sex_clean"] = df_test["sex"].map({0: 1, 1: 0}).fillna(0.5)  # 1 = male, 0 = female
    height_m = pd.to_numeric(df_test["height"], errors="coerce") / 100.0
    weight_kg = pd.to_numeric(df_test["weight"], errors="coerce")
    bmi = weight_kg / (height_m ** 2)
    df_test["bmi_clean"] = bmi.clip(lower=15.0, upper=50.0).fillna(25.0)

    # Diagnostic Labels
    scp_str = df_test["scp_codes"].astype(str)
    df_test["true_arrhythmia"] = scp_str.apply(lambda s: 1 if any(x in s for x in ["AFIB", "AFLT", "SVTAC", "PVC", "PAC"]) else 0)
    df_test["true_conduction"] = scp_str.apply(lambda s: 1 if any(x in s for x in ["CD", "LBBB", "RBBB", "1AVB", "2AVB", "3AVB", "WPW"]) else 0)
    df_test["true_infarct"] = scp_str.apply(lambda s: 1 if any(x in s for x in ["MI", "AMI", "IMI", "LMI", "PMI"]) else 0)
    df_test["true_hypertrophy"] = scp_str.apply(lambda s: 1 if any(x in s for x in ["HYP", "LVH", "RVH", "LAE", "RAE"]) else 0)
    df_test["pacemaker_flag"] = df_test["pacemaker"].fillna("").astype(str).apply(lambda s: 1 if "pace" in s.lower() else 0)

    # Load waveform tensors
    test_dir = _ROOT / "data/ptb_xl/tensors/test"
    test_files = sorted(list(test_dir.glob("*.pt")))
    signals, ecg_ids = [], []
    for p in test_files:
        sig = torch.load(p, map_location="cpu", weights_only=True).float()
        signals.append(sig)
        ecg_ids.append(int(p.stem))
    waveforms = torch.stack(signals, dim=0)

    logging.info("PTB-XL Test Cohort: %d ECGs, %d unique patients, waveforms shape %s",
                 len(df_test), df_test["patient_id"].nunique(), waveforms.shape)
    return df_test, waveforms, ecg_ids


def load_echonext_test_cohort():
    logging.info("Loading EchoNext test cohort...")
    meta_path = _ROOT / "data/echonext/echonext_metadata_100k.csv"
    echo_data, _ = load_echonext_waveforms(_ROOT / "data/echonext")
    echonext_model_root = _ROOT / "ecg_fm_integration/echonext_minimodel_repo/7-EchoNext Minimodel"
    shd_classifier = EchoNextMiniModel(echonext_model_root, "cpu")
    shd_metadata, shd_tabular, shd_labels = load_echonext_test_metadata(
        meta_path, shd_classifier.transformer_path
    )

    shd_metadata_1k = shd_metadata.iloc[:1000].copy()
    shd_tabular_1k = shd_tabular[:1000].copy()

    shd_metadata_1k["age_clean"] = pd.to_numeric(shd_metadata_1k["age_at_ecg"], errors="coerce").fillna(60.0)
    shd_metadata_1k["sex_clean"] = shd_metadata_1k["sex"].map({"Male": 1, "Female": 0, 0: 1, 1: 0}).fillna(0.5)
    shd_metadata_1k["hr_clean"] = pd.to_numeric(shd_metadata_1k["ventricular_rate"], errors="coerce").fillna(70.0)
    shd_metadata_1k["true_shd"] = pd.to_numeric(shd_metadata_1k["shd_moderate_or_greater_flag"], errors="coerce").fillna(0).astype(int)
    shd_metadata_1k["true_lvef_45"] = pd.to_numeric(shd_metadata_1k["lvef_lte_45_flag"], errors="coerce").fillna(0).astype(int)
    shd_metadata_1k["true_lvwt_13"] = pd.to_numeric(shd_metadata_1k["lvwt_gte_13_flag"], errors="coerce").fillna(0).astype(int)
    shd_metadata_1k["true_pasp_45"] = pd.to_numeric(shd_metadata_1k["pasp_gte_45_flag"], errors="coerce").fillna(0).astype(int)

    logging.info("EchoNext Test Cohort: %d patients loaded", len(shd_metadata_1k))
    return shd_metadata_1k, echo_data, shd_tabular_1k, shd_classifier


def compute_ptbxl_ground_truth_references(
    waveforms: torch.Tensor,
    ecg_ids: List[int],
    df_ptb: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64
) -> Dict[int, Dict[str, Any]]:
    logging.info("--> Computing Ground Truth Reference Biomarkers on True 12-Lead Signals...")
    semiseg_model = load_semiseg_delineator(device)

    # Load ECGFounder for reference diagnosis
    ecgfounder_repo = _ROOT / "ecg_fm_integration/ecgfounder_repo"
    tasks = load_task_names(ecgfounder_repo / "tasks.txt")
    founder_model = load_ecgfounder(ecgfounder_repo, ecgfounder_repo / "checkpoint/12_lead_ECGFounder.pth", device, len(tasks))

    references = {}
    N = len(ecg_ids)

    with torch.inference_mode():
        for i in tqdm(range(0, N, batch_size), desc="GT References"):
            b_stop = min(i + batch_size, N)
            batch_wave = waveforms[i:b_stop].to(device)
            b_ids = ecg_ids[i:b_stop]

            # 1. SemiSeg QRS on Lead II
            resamp = F.interpolate(batch_wave, size=2500, mode="linear", align_corners=False)
            std_sig = (resamp - resamp.mean(dim=-1, keepdim=True)) / (resamp.std(dim=-1, keepdim=True) + 1e-6)
            logits = semiseg_model(std_sig[:, 1:2, :])["seg_logits"]
            preds = logits.argmax(dim=1).repeat_interleave(2, dim=-1)[:, :5000].cpu().numpy()

            # 2. ECGFounder on True 12-lead
            preproc_wave = preprocess_ecgfounder(batch_wave)
            probs = torch.sigmoid(founder_model(preproc_wave)).cpu().numpy()

            wave_np = batch_wave.cpu().numpy()

            for b_idx, eid in enumerate(b_ids):
                qrs_dur = extract_qrs_duration(preds[b_idx])

                s_v1 = float(np.abs(np.min(wave_np[b_idx, 6, 1000:4000])))
                r_v5 = float(np.max(wave_np[b_idx, 10, 1000:4000]))
                sokolow = s_v1 + r_v5

                p_vec = probs[b_idx]
                p_arr = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if any(k in name.upper() for k in ["FIBRILLATION", "FLUTTER", "TACHYCARDIA", "PVC"])]))
                p_cond = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if any(k in name.upper() for k in ["BLOCK", "AVB", "WPW"])]))
                p_inf = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if "INFARCT" in name.upper()]))
                p_hyp = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if "HYPERTROPHY" in name.upper()]))

                references[eid] = {
                    "ref_qrs_dur": qrs_dur,
                    "ref_qrs_gt120": 1 if (pd.notna(qrs_dur) and qrs_dur > 120.0) else 0,
                    "ref_sokolow": sokolow,
                    "ref_lvh_gt35": 1 if sokolow > 3.5 else 0,
                    "ref_p_arr": p_arr,
                    "ref_p_cond": p_cond,
                    "ref_p_inf": p_inf,
                    "ref_p_hyp": p_hyp,
                }

    return references


def run_patient_level_extraction():
    logging.info("=" * 80)
    logging.info("STARTING 100% EMPIRICAL PATIENT-LEVEL EXTRACTION FOR ALL 55 MODELS ON CUDA:0")
    logging.info("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Compute device: %s", device)

    # 1. Load Data
    df_ptb, ptb_waveforms, ecg_ids = load_ptbxl_test_cohort()
    df_echo, echo_data, echo_tabular, echo_classifier = load_echonext_test_cohort()

    # 2. Compute GT References on PTB-XL
    ptb_gt_refs = compute_ptbxl_ground_truth_references(ptb_waveforms, ecg_ids, df_ptb, device)

    # 3. Load Evaluators
    semiseg_model = load_semiseg_delineator(device)
    ecgfounder_repo = _ROOT / "ecg_fm_integration/ecgfounder_repo"
    tasks = load_task_names(ecgfounder_repo / "tasks.txt")
    founder_model = load_ecgfounder(ecgfounder_repo, ecgfounder_repo / "checkpoint/12_lead_ECGFounder.pth", device, len(tasks))

    # 4. Roster: FULL 55 Models
    roster = build_55_model_roster()
    logging.info("Loaded complete benchmark roster: %d models across 4 tracks", len(roster))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_models = set()
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH, timeout=30) as con:
                tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                if "ptbxl_patient_observations" in tables:
                    res = con.execute("SELECT DISTINCT model_id FROM ptbxl_patient_observations").fetchall()
                    done_models = {r[0] for r in res}
                    logging.info("Found %d models already extracted in database.", len(done_models))
        except Exception as e:
            logging.warning("Error checking existing DB tables: %s", e)

    batch_size = 64

    for m_idx, item in enumerate(roster, 1):
        mid = item["model_id"]
        track = item["track"]
        ckpt_path = item["checkpoint_path"]
        is_ephemeral = item.get("is_ephemeral", False)
        fam = map_track_to_family(track)

        if mid in done_models:
            logging.info("[%d/%d] Model %s already in database, skipping.", m_idx, len(roster), mid)
            continue

        logging.info("-" * 80)
        logging.info("[%d/%d] [%s] Extracting Patient Observations: %s", m_idx, len(roster), fam, mid)
        logging.info("-" * 80)

        # On-demand materialization for ephemeral spatial grid checkpoints
        if is_ephemeral:
            logging.info("Materializing ephemeral checkpoint for %s...", mid)
            res = subprocess.run(
                [sys.executable, str(_ROOT / "scripts/onelead_checkpoint_store.py"), "materialize", mid],
                capture_output=True,
                text=True
            )
            if res.returncode != 0:
                logging.error("Failed to materialize %s: %s", mid, res.stderr)
                continue
            ckpt_path = Path(res.stdout.strip())

        t0 = time.time()
        try:
            reconstructor = Unified1LeadReconstructor(mid, ckpt_path, device)

            # -----------------------------------------------------------------
            # PTB-XL Extraction (N = 2,198 ECGs)
            # -----------------------------------------------------------------
            ptb_records_model = []
            N_ptb = len(ecg_ids)
            with torch.inference_mode():
                for i in range(0, N_ptb, batch_size):
                    b_stop = min(i + batch_size, N_ptb)
                    batch_wave = ptb_waveforms[i:b_stop].to(device)
                    b_ids = ecg_ids[i:b_stop]

                    recon = reconstructor.reconstruct(batch_wave)

                    resamp_r = F.interpolate(recon, size=2500, mode="linear", align_corners=False)
                    std_r = (resamp_r - resamp_r.mean(dim=-1, keepdim=True)) / (resamp_r.std(dim=-1, keepdim=True) + 1e-6)
                    logits_r = semiseg_model(std_r[:, 1:2, :])["seg_logits"]
                    preds_r = logits_r.argmax(dim=1).repeat_interleave(2, dim=-1)[:, :5000].cpu().numpy()

                    preproc_r = preprocess_ecgfounder(recon)
                    probs_r = torch.sigmoid(founder_model(preproc_r)).cpu().numpy()

                    true_np = batch_wave.cpu().numpy()
                    recon_np = recon.cpu().numpy()

                    for b_idx, eid in enumerate(b_ids):
                        ptb_row = df_ptb.loc[eid]
                        pid = ptb_row["patient_id"]
                        gt = ptb_gt_refs[eid]

                        qrs_dur_r = extract_qrs_duration(preds_r[b_idx])
                        qrs_err = abs(qrs_dur_r - gt["ref_qrs_dur"]) if (pd.notna(qrs_dur_r) and pd.notna(gt["ref_qrs_dur"])) else np.nan
                        qrs_signed_err = (qrs_dur_r - gt["ref_qrs_dur"]) if (pd.notna(qrs_dur_r) and pd.notna(gt["ref_qrs_dur"])) else np.nan
                        concord_cd = 1 if (pd.notna(qrs_dur_r) and pd.notna(gt["ref_qrs_dur"]) and (qrs_dur_r > 120.0) == (gt["ref_qrs_dur"] > 120.0)) else 0

                        s_v1_r = float(np.abs(np.min(recon_np[b_idx, 6, 1000:4000])))
                        r_v5_r = float(np.max(recon_np[b_idx, 10, 1000:4000]))
                        sokolow_r = s_v1_r + r_v5_r
                        sokolow_err = abs(sokolow_r - gt["ref_sokolow"])
                        concord_lvh = 1 if (sokolow_r > 3.5) == (gt["ref_sokolow"] > 3.5) else 0

                        def get_r(lead_idx):
                            yt = true_np[b_idx, lead_idx]
                            yr = recon_np[b_idx, lead_idx]
                            yt_c = yt - np.mean(yt)
                            yr_c = yr - np.mean(yr)
                            denom = np.sqrt(np.sum(yt_c ** 2) * np.sum(yr_c ** 2)) + 1e-8
                            return float(np.sum(yt_c * yr_c) / denom)

                        r_v3 = get_r(8)
                        r_v6 = get_r(11)
                        r_ii = get_r(1)
                        r_chest = float(np.mean([get_r(k) for k in range(6, 12)]))

                        p_vec = probs_r[b_idx]
                        p_arr_r = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if any(k in name.upper() for k in ["FIBRILLATION", "FLUTTER", "TACHYCARDIA", "PVC"])]))
                        p_cond_r = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if any(k in name.upper() for k in ["BLOCK", "AVB", "WPW"])]))
                        p_inf_r = float(np.mean([p_vec[t] for t, name in enumerate(tasks) if "INFARCT" in name.upper()]))

                        concord_arr = 1 if (p_arr_r > 0.5) == ptb_row["true_arrhythmia"] else 0
                        concord_cond = 1 if (p_cond_r > 0.5) == ptb_row["true_conduction"] else 0
                        concord_inf = 1 if (p_inf_r > 0.5) == ptb_row["true_infarct"] else 0

                        ptb_records_model.append({
                            "model_id": mid,
                            "architecture_family": fam,
                            "patient_id": pid,
                            "ecg_id": eid,
                            "age": ptb_row["age_clean"],
                            "sex": ptb_row["sex_clean"],
                            "bmi": ptb_row["bmi_clean"],
                            "pacemaker": ptb_row["pacemaker_flag"],
                            "true_arrhythmia": ptb_row["true_arrhythmia"],
                            "true_conduction": ptb_row["true_conduction"],
                            "true_infarct": ptb_row["true_infarct"],
                            "true_hypertrophy": ptb_row["true_hypertrophy"],
                            "ref_qrs_dur": gt["ref_qrs_dur"],
                            "recon_qrs_dur": qrs_dur_r,
                            "qrs_error_ms": qrs_err,
                            "qrs_signed_error_ms": qrs_signed_err,
                            "ref_sokolow_mv": gt["ref_sokolow"],
                            "recon_sokolow_mv": sokolow_r,
                            "sokolow_error_mv": sokolow_err,
                            "r_v3": r_v3,
                            "r_v6": r_v6,
                            "r_ii": r_ii,
                            "r_chest": r_chest,
                            "recon_p_arr": p_arr_r,
                            "recon_p_cond": p_cond_r,
                            "recon_p_inf": p_inf_r,
                            "concord_conduction_delay": concord_cd,
                            "concord_sokolow_lvh": concord_lvh,
                            "concord_arrhythmia": concord_arr,
                            "concord_conduction": concord_cond,
                            "concord_infarct": concord_inf,
                        })

            # -----------------------------------------------------------------
            # EchoNext Extraction (N = 1,000 ECGs, 500Hz)
            # -----------------------------------------------------------------
            echo_records_model = []
            N_echo = 1000
            with torch.inference_mode():
                for i in range(0, N_echo, batch_size):
                    b_stop = min(i + batch_size, N_echo)
                    batch_np = echo_data.batch(i, b_stop)
                    batch_wave = torch.from_numpy(batch_np).float().to(device)

                    recon_echo = reconstructor.reconstruct(batch_wave)
                    probs_echo = echo_classifier.predict_reconstruction_500hz(
                        recon_echo,
                        echo_tabular[i:b_stop]
                    )

                    for b_idx in range(b_stop - i):
                        global_idx = i + b_idx
                        erow = df_echo.iloc[global_idx]
                        pkey = erow["patient_key"]
                        ekey = erow["ecg_key"]

                        p_shd = float(probs_echo[b_idx, 11])   # task 11: shd_moderate_or_greater
                        p_ef = float(probs_echo[b_idx, 0])     # task 0: lvef_lte_45
                        p_wt = float(probs_echo[b_idx, 1])     # task 1: lvwt_gte_13
                        p_ph = float(probs_echo[b_idx, 9])     # task 9: pasp_gte_45

                        echo_records_model.append({
                            "model_id": mid,
                            "architecture_family": fam,
                            "patient_key": pkey,
                            "ecg_key": ekey,
                            "age": erow["age_clean"],
                            "sex": erow["sex_clean"],
                            "ventricular_rate": erow["hr_clean"],
                            "true_shd": erow["true_shd"],
                            "true_lvef_45": erow["true_lvef_45"],
                            "true_lvwt_13": erow["true_lvwt_13"],
                            "true_pasp_45": erow["true_pasp_45"],
                            "recon_p_shd": p_shd,
                            "recon_p_lvef_45": p_ef,
                            "recon_p_lvwt_13": p_wt,
                            "recon_p_pasp_45": p_ph,
                            "concord_shd": 1 if (p_shd > 0.5) == erow["true_shd"] else 0,
                            "concord_lvef_45": 1 if (p_ef > 0.5) == erow["true_lvef_45"] else 0,
                            "concord_lvwt_13": 1 if (p_wt > 0.5) == erow["true_lvwt_13"] else 0,
                            "concord_pasp_45": 1 if (p_ph > 0.5) == erow["true_pasp_45"] else 0,
                        })

            # Persist model batch immediately to SQLite
            df_ptb_m = pd.DataFrame(ptb_records_model)
            df_echo_m = pd.DataFrame(echo_records_model)
            with sqlite3.connect(DB_PATH, timeout=60) as con:
                df_ptb_m.to_sql("ptbxl_patient_observations", con, if_exists="append", index=False)
                df_echo_m.to_sql("echonext_patient_observations", con, if_exists="append", index=False)

            dt_sec = time.time() - t0
            logging.info("  --> Saved %s in %.1fs: PTB-XL N=%d, EchoNext N=%d", mid, dt_sec, len(df_ptb_m), len(df_echo_m))

        except Exception as e:
            logging.exception("Error extracting patient observations for %s: %s", mid, e)
        finally:
            if is_ephemeral and ckpt_path.exists():
                logging.info("Evicting ephemeral checkpoint %s to reclaim disk space...", ckpt_path.name)
                ckpt_path.unlink(missing_ok=True)

    # Export full SQLite tables to parquet and create indexes
    with sqlite3.connect(DB_PATH, timeout=60) as con:
        con.execute("CREATE INDEX IF NOT EXISTS idx_ptb_pid_mid ON ptbxl_patient_observations (patient_id, model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ptb_mid ON ptbxl_patient_observations (model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_echo_pkey_mid ON echonext_patient_observations (patient_key, model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_echo_mid ON echonext_patient_observations (model_id)")

        df_ptb_all = pd.read_sql("SELECT * FROM ptbxl_patient_observations", con)
        df_echo_all = pd.read_sql("SELECT * FROM echonext_patient_observations", con)

    df_ptb_all.to_parquet(OUT_DIR / "ptbxl_patient_observations.parquet", index=False)
    df_echo_all.to_parquet(OUT_DIR / "echonext_patient_observations.parquet", index=False)

    logging.info("=" * 80)
    logging.info("PATIENT-LEVEL EXTRACTION FOR ALL 55 MODELS COMPLETE:")
    logging.info("  PTB-XL Patient Observations: %d rows (%d models)", len(df_ptb_all), df_ptb_all["model_id"].nunique())
    logging.info("  EchoNext Patient Observations: %d rows (%d models)", len(df_echo_all), df_echo_all["model_id"].nunique())
    logging.info("=" * 80)


if __name__ == "__main__":
    run_patient_level_extraction()
