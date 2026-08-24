#!/usr/bin/env python3
"""Sunnybrook all-feature external benchmark for the exact baseline VAE.

This evaluator measures how much clinically relevant information is preserved
when reconstructing 12 leads from the observed regime `II, V1, V5`.

It produces two layers of analysis:
1. Direct zero-shot feature preservation for waveform-extractable globals.
2. All-feature downstream prediction using fixed linear probes under LOOCV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.reconstruction.unified_latents.engineering.common import LEAD_NAMES, write_json
from src.reconstruction.unified_latents.engineering.vae import WearECGVAE
from src.reconstruction.unified_latents.engineering.vae_fm import WearECGFMVAE


DEFAULT_CHECKPOINT = (
    "/home/mithunmanivannan/checkpoints/ul_ecg/"
    "engineering_wearecg_exact_II-V1-V5_bs64_lf1.5_ep10_canonical_baseline/"
    "ul_ecp_best.pt"
)
DEFAULT_OUTPUT_ROOT = "/home/mithunmanivannan/reports/sunnybrook_all_feature_baseline"
LEAD_ORDER = list(LEAD_NAMES)
LEAD_TO_INDEX = {lead: idx for idx, lead in enumerate(LEAD_ORDER)}
SUNNYBROOK_UV_PER_BIT = 5.0
TEXT_LABEL_RULES = {
    "conduction_any": [
        "bundle branch block",
        "intraventricular conduction delay",
        "av block",
        "right bundle branch block",
        "left bundle branch block",
        "incomplete right bundle branch block",
    ],
    "paced_any": ["paced", "ventricular-paced", "v-paced", "ventricular paced"],
    "afib_flutter": ["atrial fibrillation", "afib", "atrial flutter", "flutter"],
    "axis_dev": ["left axis deviation", "right axis deviation"],
    "ischemia_st": ["ischemia", "st elevation", "st depression", "repol abnrm"],
    "low_voltage": ["low voltage"],
    "atrial_abn": ["left atrial enlargement", "probable left atrial enlargement"],
    "sinus_non_normal": ["sinus bradycardia", "sinus tachycardia", "sinus arrhythmia"],
}
DIAG_CODE_FAMILY_MAP: dict[str, str] = {
    "SR": "rhythm",
    "SB": "rhythm",
    "ST": "rhythm",
    "SA": "rhythm",
    "AFIB0": "rhythm",
    "AFLT2": "rhythm",
    "VPACEF": "rhythm",
    "VPACEC": "rhythm",
    "NIVCD": "conduction_axis",
    "RBBB": "conduction_axis",
    "IRBBB": "conduction_axis",
    "AXL": "conduction_axis",
    "RAD": "conduction_axis",
    "IMIC": "ischemia_infarct_repolarization",
    "MSTEA": "ischemia_infarct_repolarization",
    "T0DI": "ischemia_infarct_repolarization",
    "T0IN": "ischemia_infarct_repolarization",
    "T1IN": "ischemia_infarct_repolarization",
    "T6AN": "ischemia_infarct_repolarization",
    "EREPOL": "ischemia_infarct_repolarization",
    "REPILA": "ischemia_infarct_repolarization",
    "SD0IN": "ischemia_infarct_repolarization",
    "QMML": "ischemia_infarct_repolarization",
    "LVOLF": "chamber_voltage_technical",
    "LVOLT": "chamber_voltage_technical",
    "LAE": "chamber_voltage_technical",
    "PLAE": "chamber_voltage_technical",
    "MISLDS": "chamber_voltage_technical",
    "ET": "chamber_voltage_technical",
}
DIRECT_FEATURES = ["heart_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc", "qrs_axis"]
GLOBAL_CANONICAL_MAP: dict[str, list[tuple[str, str]]] = {
    "age": [("master", "age"), ("features", "age_years")],
    "heart_rate": [("master", "heart_rate"), ("features", "heart_rate_bpm")],
    "pr_interval": [("master", "pr_interval"), ("features", "pr_interval_ms")],
    "qrs_duration": [("master", "qrs_duration"), ("features", "qrs_duration_ms")],
    "qt_interval": [("master", "qt_interval"), ("features", "qt_interval_ms")],
    "qtc": [("master", "qtc"), ("extra", "QTc")],
    "qrs_axis": [("master", "qrs_axis"), ("features", "qrs_axis_deg")],
    "t_axis": [("master", "t_axis"), ("features", "t_wave_axis_deg")],
    "p_axis": [("master", "p_axis"), ("extra", "P_Axis")],
}
FEATURE_ADMIN_COLUMNS = {
    "file",
    "encounter_id_hash",
    "machine_model_hash",
    "processing_timestamp",
    "source_md5",
    "sampling_rate_hz",
    "lowpass_hz",
    "hipass_hz",
    "sex_code",
    "pacemaker_status",
    "encounter_id",
    "machine_model",
    "patient_id",
}
FEATURE_ADMIN_PREFIXES = ("is_missing_",)
EXTRA_METADATA_COLUMNS = {"file", "QA_Action", "HPF", "LPF", "Notch", "Device", "Interpretation"}


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true")
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sunnybrook all-feature benchmark for exact baseline VAE.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-family", choices=["auto", "exact_vae", "fm_vae"], default="auto")
    parser.add_argument("--sunnybrook-dir", default="/home/mithunmanivannan/data/sunnybrook")
    parser.add_argument("--master-csv", default="/home/mithunmanivannan/data/sunnybrook_master_hyperfeatures.csv")
    parser.add_argument("--features-csv", default="/home/mithunmanivannan/data/sunnybrook_features.csv")
    parser.add_argument("--extra-csv", default="/home/mithunmanivannan/data/sunnybrook_extra_metadata.csv")
    parser.add_argument("--obs-leads", default="II,V1,V5")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    add_bool_arg(parser, "use-amp", True)
    return parser.parse_args()


def parse_obs_leads(obs_leads: str) -> list[int]:
    leads = [lead.strip() for lead in obs_leads.split(",") if lead.strip()]
    if not leads:
        raise ValueError("Observed leads must not be empty.")
    bad = [lead for lead in leads if lead not in LEAD_TO_INDEX]
    if bad:
        raise ValueError(f"Unsupported observed leads: {bad}")
    return [LEAD_TO_INDEX[lead] for lead in leads]


def load_sunnybrook_record(xml_path: Path, target_len: int = 5000) -> np.ndarray:
    import sierraecg

    record = sierraecg.read_file(str(xml_path))
    signal_map = {lead.label: lead.samples for lead in record.leads}
    if not all(lead in signal_map for lead in LEAD_ORDER):
        raise RuntimeError(f"Missing leads in {xml_path.name}")
    sig = np.stack([signal_map[lead] for lead in LEAD_ORDER], axis=0).astype(np.float32)
    # Sierra ECG samples are stored as ADC counts. Sunnybrook exports use
    # 5 microvolts per bit, so convert counts -> uV -> mV before any modeling.
    sig_mv = (sig * SUNNYBROOK_UV_PER_BIT) / 1000.0
    if sig_mv.shape[1] < target_len:
        sig_mv = np.pad(sig_mv, ((0, 0), (0, target_len - sig_mv.shape[1])))
    else:
        sig_mv = sig_mv[:, :target_len]
    return sig_mv.astype(np.float32)


def load_exact_baseline(checkpoint_path: str, device: torch.device) -> tuple[WearECGVAE, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = WearECGVAE(
        target_len=int(ckpt.get("target_len", 5000)),
        beta_kl=float(ckpt.get("beta_kl", 1e-4)),
    )
    if "encoder_state_dict" in ckpt and "decoder_state_dict" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        model.decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model, ckpt


def load_fm_vae(checkpoint_path: str, device: torch.device) -> tuple[WearECGFMVAE, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    encoder_in_channels = int(state["encoder.blocks.0.weight"].shape[1])
    inferred_mask_aware = encoder_in_channels == 24
    inferred_split_latent = any(key.startswith("encoder.local_head.") for key in state) and any(
        key.startswith("encoder.global_head.") for key in state
    )
    inferred_global_channels = int(state["encoder.global_head.2.weight"].shape[0] // 2) if inferred_split_latent else int(
        ckpt.get("global_latent_channels", 2)
    )
    inferred_local_channels = int(state["encoder.local_head.2.weight"].shape[0] // 2) if inferred_split_latent else int(
        ckpt.get("local_latent_channels", 2)
    )
    model = WearECGFMVAE(
        fm_checkpoint_path=str(ckpt["fm_checkpoint"]),
        latent_channels=int(ckpt.get("latent_channels", 4)),
        target_len=int(ckpt.get("target_len", 5000)),
        beta_kl=float(ckpt.get("beta_kl", 1e-4)),
        missing_lead_weight=float(ckpt.get("missing_lead_weight", 1.0)),
        fm_loss_weight=float(ckpt.get("fm_loss_weight", 1e-2)),
        fm_cosine_mix=float(ckpt.get("fm_cosine_mix", 0.5)),
        use_decoder_conditioning=bool(ckpt.get("use_decoder_conditioning", ckpt.get("fm_decoder_conditioning", False))),
        fm_cond_drop_prob=float(ckpt.get("fm_cond_drop_prob", 0.0)),
        use_latent_alignment=bool(ckpt.get("use_latent_alignment", ckpt.get("fm_latent_align", False))),
        latent_align_weight=float(ckpt.get("latent_align_weight", 1e-3)),
        mask_aware_encoder=bool(ckpt.get("mask_aware_encoder", inferred_mask_aware)),
        split_latent=bool(ckpt.get("split_latent", inferred_split_latent)),
        global_latent_channels=int(ckpt.get("global_latent_channels", inferred_global_channels)),
        local_latent_channels=int(ckpt.get("local_latent_channels", inferred_local_channels)),
        use_multi_scale_align=bool(ckpt.get("use_multi_scale_align", ckpt.get("fm_multi_scale_align", False))),
        multi_scale_align_weight=float(ckpt.get("multi_scale_align_weight", 1e-1)),
    )
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("fm_model.backbone.",)
    allowed_missing_exact = {
        "fm_model.token_norm.weight",
        "fm_model.token_norm.bias",
        "encoder.output_head.0.weight",
        "encoder.output_head.0.bias",
        "encoder.output_head.1.weight",
        "encoder.output_head.1.bias",
    }
    bad_missing = [
        key for key in incompatible.missing_keys
        if key not in allowed_missing_exact and not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys while loading FM checkpoint: {bad_missing}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading FM checkpoint: {incompatible.unexpected_keys}")
    model.to(device).eval()
    return model, ckpt


def load_reconstruction_model(
    checkpoint_path: str,
    device: torch.device,
    model_family: str,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    inferred_family = "fm_vae" if ckpt.get("model_family") == "fm_vae" else "exact_vae"
    chosen_family = inferred_family if model_family == "auto" else model_family

    if chosen_family == "fm_vae":
        model, ckpt = load_fm_vae(checkpoint_path, device)
    else:
        model, ckpt = load_exact_baseline(checkpoint_path, device)
    return model, ckpt, chosen_family


def reconstruct_sources(
    model: torch.nn.Module,
    records: list[Path],
    obs_indices: list[int],
    device: torch.device,
    use_amp: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, list[np.ndarray]] = {"orig12": [], "recon12": [], "obs3": []}
    obs_leads = [LEAD_ORDER[idx] for idx in obs_indices]
    autocast_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        for xml_path in tqdm(records, desc="Reconstructing Sunnybrook"):
            signal_mv = load_sunnybrook_record(xml_path)
            x = torch.from_numpy(signal_mv).unsqueeze(0).to(device=device, dtype=torch.float32)
            lead_idx_tensor = torch.tensor([obs_indices], device=device)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled, dtype=torch.bfloat16):
                out = model.impute_from_regressor(x, lead_indices=lead_idx_tensor)
            recon = out["y_pred"].squeeze(0).detach().cpu().float().numpy()
            obs = signal_mv[obs_indices].copy()

            sources["orig12"].append(signal_mv.copy())
            sources["recon12"].append(recon)
            sources["obs3"].append(obs)
            rows.append(
                {
                    "file": xml_path.name,
                    "record_id": xml_path.stem,
                    "orig12_shape": "12x5000",
                    "recon12_shape": "12x5000",
                    "obs3_shape": f"{len(obs_indices)}x5000",
                    "obs_leads": ",".join(obs_leads),
                    "orig12_available_leads": ",".join(LEAD_ORDER),
                    "recon12_available_leads": ",".join(LEAD_ORDER),
                    "obs3_available_leads": ",".join(obs_leads),
                }
            )

    arrays = {key: np.stack(value, axis=0) for key, value in sources.items()}
    return pd.DataFrame(rows), arrays


def normalize_index(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    out["file"] = out["file"].astype(str)
    out = out.set_index("file").sort_index()
    out.attrs["source_name"] = source_name
    return out


def choose_series(
    frames: dict[str, pd.DataFrame],
    choices: list[tuple[str, str]],
) -> tuple[pd.Series, str, str]:
    for frame_name, column in choices:
        frame = frames[frame_name]
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.notna().sum() > 0:
            return series, frame_name, column
    first_frame, first_column = choices[0]
    return pd.Series(index=frames[first_frame].index, dtype=float), first_frame, first_column


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_diag_codes(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [code.strip() for code in re.split(r"[;,|]", str(value)) if code.strip()]


def mine_text_label(text: str | float | None, patterns: list[str]) -> int:
    if pd.isna(text):
        return 0
    text_l = str(text).lower()
    return int(any(pattern in text_l for pattern in patterns))


def infer_group(name: str) -> str:
    if name in {"age", "heart_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc", "qrs_axis", "t_axis", "p_axis"}:
        return "global_intervals_axes"
    if name.endswith("_amp"):
        return "leadwise_amplitudes"
    if name.endswith("_dur"):
        return "leadwise_durations"
    if "_st_" in name or name.endswith("_st_slope"):
        return "st_metrics"
    if name.startswith("diag_code__"):
        return "structured_diagnosis"
    if name in TEXT_LABEL_RULES:
        return "text_mined_labels"
    if name.endswith("_measured"):
        return "signal_integrity"
    return "other"


def infer_diag_code_family(token: str) -> str:
    return DIAG_CODE_FAMILY_MAP.get(token, "other_diag_code")


def diag_code_support_tier(values: pd.Series) -> str:
    valid = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    pos = int(valid.sum())
    neg = int((1 - valid).sum())
    if pos == 0:
        return "absent"
    if pos == 1:
        return "singleton_audit"
    if pos >= 2 and neg >= 2:
        return "supported"
    return "unsupported_class_imbalance"


def support_tier_for_series(series: pd.Series, target_type: str) -> str:
    if target_type == "classification":
        valid = series.dropna().astype(int)
        pos = int(valid.sum())
        neg = int((1 - valid).sum())
        if pos < 2 or neg < 2:
            return "unsupported_class_imbalance"
        if pos <= 3 or neg <= 3:
            return "exploratory"
        return "supported"
    if target_type == "regression":
        valid = series.dropna()
        unique = int(valid.nunique(dropna=True))
        if len(valid) < 4 or unique < 2:
            return "unsupported_regression"
        if len(valid) < 8 or unique < 5:
            return "exploratory"
        return "supported"
    return "metadata_only"


def build_target_catalog(
    master_csv: str,
    features_csv: str,
    extra_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {
        "master": normalize_index(pd.read_csv(master_csv), "master"),
        "features": normalize_index(pd.read_csv(features_csv), "features"),
        "extra": normalize_index(pd.read_csv(extra_csv), "extra"),
    }
    index = frames["master"].index
    target_columns: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    canonical_taken = set(GLOBAL_CANONICAL_MAP)

    for canonical_name, choices in GLOBAL_CANONICAL_MAP.items():
        series, source_name, source_column = choose_series(frames, choices)
        numeric = safe_numeric(series)
        target_columns[canonical_name] = numeric
        rows.append(
            {
                "canonical_name": canonical_name,
                "source_name": source_name,
                "source_column": source_column,
                "target_type": "regression",
                "missing_count": int(numeric.isna().sum()),
                "unique_count": int(numeric.nunique(dropna=True)),
                "support_tier": support_tier_for_series(numeric, "regression"),
                "group": infer_group(canonical_name),
                "notes": "",
            }
        )

    master = frames["master"]
    for column in master.columns:
        if column in {"diag_codes"} or column in canonical_taken:
            continue
        if column == "file":
            rows.append(
                {
                    "canonical_name": "file",
                    "source_name": "master",
                    "source_column": "file",
                    "target_type": "metadata_only",
                    "missing_count": 0,
                    "unique_count": int(master.index.nunique()),
                    "support_tier": "metadata_only",
                    "group": "metadata",
                    "notes": "record identifier",
                }
            )
            continue

        series = master[column]
        if pd.api.types.is_bool_dtype(series):
            values = series.astype(int)
            target_type = "classification"
        else:
            numeric = safe_numeric(series)
            if numeric.notna().sum() > 0:
                values = numeric
                target_type = "regression"
            else:
                rows.append(
                    {
                        "canonical_name": column,
                        "source_name": "master",
                        "source_column": column,
                        "target_type": "unsupported",
                        "missing_count": int(series.isna().sum()),
                        "unique_count": int(series.nunique(dropna=True)),
                        "support_tier": "unsupported_non_numeric",
                        "group": infer_group(column),
                        "notes": "non-numeric target",
                    }
                )
                continue

        target_columns[column] = values
        rows.append(
            {
                "canonical_name": column,
                "source_name": "master",
                "source_column": column,
                "target_type": target_type,
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "support_tier": support_tier_for_series(values, target_type),
                "group": infer_group(column),
                "notes": "",
            }
        )

    diag_tokens = sorted({token for value in master.get("diag_codes", pd.Series(index=index, dtype=object)) for token in parse_diag_codes(value)})
    for token in diag_tokens:
        name = f"diag_code__{token}"
        values = master["diag_codes"].apply(lambda value: int(token in parse_diag_codes(value))).astype(int)
        tier = diag_code_support_tier(values)
        target_columns[name] = values
        rows.append(
            {
                "canonical_name": name,
                "source_name": "master",
                "source_column": "diag_codes",
                "target_type": "classification",
                "missing_count": 0,
                "unique_count": int(values.nunique(dropna=True)),
                "support_tier": tier,
                "group": infer_group(name),
                "notes": "structured diag_codes one-vs-rest label",
            }
        )

    interpretation = frames["extra"]["Interpretation"] if "Interpretation" in frames["extra"].columns else pd.Series(index=index, dtype=object)
    for name, patterns in TEXT_LABEL_RULES.items():
        values = interpretation.apply(lambda text: mine_text_label(text, patterns)).astype(int)
        target_columns[name] = values
        rows.append(
            {
                "canonical_name": name,
                "source_name": "extra",
                "source_column": "Interpretation",
                "target_type": "classification",
                "missing_count": 0,
                "unique_count": int(values.nunique(dropna=True)),
                "support_tier": support_tier_for_series(values, "classification"),
                "group": infer_group(name),
                "notes": "rule-based text label",
            }
        )

    diag_meta = master.get("diag_codes", pd.Series(index=index, dtype=object))
    target_columns["diag_codes"] = diag_meta
    rows.append(
        {
            "canonical_name": "diag_codes",
            "source_name": "master",
            "source_column": "diag_codes",
            "target_type": "metadata_only",
            "missing_count": int(diag_meta.isna().sum()),
            "unique_count": int(diag_meta.nunique(dropna=True)),
            "support_tier": "metadata_only",
            "group": "metadata",
            "notes": "structured source for diagnosis labels",
        }
    )

    target_columns["Interpretation"] = interpretation
    rows.append(
        {
            "canonical_name": "Interpretation",
            "source_name": "extra",
            "source_column": "Interpretation",
            "target_type": "metadata_only",
            "missing_count": int(interpretation.isna().sum()),
            "unique_count": int(interpretation.nunique(dropna=True)),
            "support_tier": "metadata_only",
            "group": "metadata",
            "notes": "free-text interpretation source",
        }
    )

    target_values = pd.DataFrame(target_columns, index=index)
    catalog = pd.DataFrame(rows).sort_values(["target_type", "group", "canonical_name"]).reset_index(drop=True)
    return catalog, target_values


def make_target_catalog(
    master_csv: str,
    features_csv: str,
    extra_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog, target_values = build_target_catalog(master_csv, features_csv, extra_csv)
    unsupported_rows = []

    raw_features = pd.read_csv(features_csv)
    for column in raw_features.columns:
        if column in FEATURE_ADMIN_COLUMNS or column.startswith(FEATURE_ADMIN_PREFIXES):
            unsupported_rows.append(
                {
                    "canonical_name": column,
                    "source_name": "features",
                    "source_column": column,
                    "target_type": "metadata_only",
                    "missing_count": int(raw_features[column].isna().sum()),
                    "unique_count": int(raw_features[column].nunique(dropna=True)),
                    "support_tier": "metadata_only",
                    "group": "metadata",
                    "notes": "excluded administrative feature",
                }
            )

    raw_extra = pd.read_csv(extra_csv)
    for column in raw_extra.columns:
        if column in EXTRA_METADATA_COLUMNS:
            continue
        unsupported_rows.append(
            {
                "canonical_name": column,
                "source_name": "extra",
                "source_column": column,
                "target_type": "metadata_only",
                "missing_count": int(raw_extra[column].isna().sum()),
                "unique_count": int(raw_extra[column].nunique(dropna=True)),
                "support_tier": "metadata_only",
                "group": "metadata",
                "notes": "excluded metadata field",
            }
        )

    if unsupported_rows:
        catalog = pd.concat([catalog, pd.DataFrame(unsupported_rows)], ignore_index=True, axis=0)
        catalog = catalog.drop_duplicates(subset=["canonical_name", "source_name", "source_column"], keep="first")
        catalog = catalog.sort_values(["target_type", "group", "canonical_name"]).reset_index(drop=True)
    return catalog, target_values


def build_diag_code_catalog(target_values: pd.DataFrame) -> pd.DataFrame:
    diag_columns = sorted([col for col in target_values.columns if col.startswith("diag_code__")])
    n_records = len(target_values.index)
    rows: list[dict[str, Any]] = []
    for col in diag_columns:
        token = col.replace("diag_code__", "", 1)
        values = pd.to_numeric(target_values[col], errors="coerce").fillna(0).astype(int)
        positive_files = target_values.index[values == 1].tolist()
        pos = int(values.sum())
        tier = diag_code_support_tier(values)
        rows.append(
            {
                "target": col,
                "diag_code": token,
                "positive_count": pos,
                "negative_count": int((1 - values).sum()),
                "prevalence": float(pos / n_records) if n_records else np.nan,
                "support_tier": tier,
                "included_in_scored_metrics": bool(tier == "supported"),
                "family_group": infer_diag_code_family(token),
                "record_files": "|".join(positive_files),
                "notes": "literal diag_codes token",
            }
        )
    return pd.DataFrame(rows).sort_values(["support_tier", "positive_count", "diag_code"], ascending=[True, False, True]).reset_index(drop=True)


def build_full_cohort_diag_code_audit(
    files: list[str],
    flattened: dict[str, np.ndarray],
    target_values: pd.DataFrame,
    diag_catalog: pd.DataFrame,
    pca_dim: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    singleton_catalog = diag_catalog[diag_catalog["support_tier"] == "singleton_audit"].copy()
    n_records = len(files)
    for row in singleton_catalog.itertuples(index=False):
        y = pd.to_numeric(target_values[row.target], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) != 1:
            continue
        positive_idx = int(pos_idx[0])
        rec: dict[str, Any] = {
            "target": row.target,
            "diag_code": row.diag_code,
            "family_group": row.family_group,
            "positive_file": files[positive_idx],
            "positive_count": int(row.positive_count),
            "support_tier": row.support_tier,
            "audit_method": "full_cohort_logistic_rank",
            "notes": "Audit-only ranking fit on the full Sunnybrook cohort; not a held-out metric.",
        }
        for source_name, matrix in flattened.items():
            if len(np.unique(y)) < 2:
                rec[f"{source_name}_positive_score"] = np.nan
                rec[f"{source_name}_positive_rank"] = np.nan
                rec[f"{source_name}_positive_percentile"] = np.nan
                rec[f"{source_name}_positive_top3"] = False
                continue
            scaler = StandardScaler()
            x = scaler.fit_transform(matrix)
            k = max(1, min(pca_dim, x.shape[0] - 2, x.shape[1]))
            if k < x.shape[1]:
                pca = PCA(n_components=k, svd_solver="auto", random_state=0)
                x = pca.fit_transform(x)
            model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=0)
            model.fit(x, y)
            scores = model.predict_proba(x)[:, 1]
            positive_score = float(scores[positive_idx])
            rank_desc = int(1 + np.sum(scores > positive_score))
            rec[f"{source_name}_positive_score"] = positive_score
            rec[f"{source_name}_positive_rank"] = rank_desc
            rec[f"{source_name}_positive_percentile"] = float((n_records - rank_desc + 1) / n_records) if n_records else np.nan
            rec[f"{source_name}_positive_top3"] = bool(rank_desc <= 3)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["family_group", "diag_code"]).reset_index(drop=True) if rows else pd.DataFrame()


def build_diag_code_metrics(
    classification_results: pd.DataFrame,
    diag_catalog: pd.DataFrame,
) -> pd.DataFrame:
    if classification_results.empty:
        return pd.DataFrame()
    scored = diag_catalog[diag_catalog["included_in_scored_metrics"]].copy()
    if scored.empty:
        return pd.DataFrame()
    result = classification_results.merge(scored, on="target", how="inner")
    for col in ["diag_code", "family_group", "positive_count", "prevalence", "support_tier"]:
        if col not in result.columns:
            left = f"{col}_x"
            right = f"{col}_y"
            if left in result.columns and right in result.columns:
                result[col] = result[left].combine_first(result[right])
            elif left in result.columns:
                result[col] = result[left]
            elif right in result.columns:
                result[col] = result[right]
    cols = [
        "target",
        "diag_code",
        "family_group",
        "positive_count",
        "prevalence",
        "support_tier",
        "orig12_n",
        "orig12_prevalence",
        "orig12_auroc",
        "orig12_average_precision",
        "orig12_balanced_accuracy",
        "recon12_n",
        "recon12_prevalence",
        "recon12_auroc",
        "recon12_average_precision",
        "recon12_balanced_accuracy",
        "obs3_n",
        "obs3_prevalence",
        "obs3_auroc",
        "obs3_average_precision",
        "obs3_balanced_accuracy",
        "recon_minus_obs3",
        "orig12_minus_recon",
        "recovery_ratio",
    ]
    return result[cols].sort_values(["family_group", "diag_code"]).reset_index(drop=True)


def summarize_diag_code_families(diag_metrics: pd.DataFrame) -> pd.DataFrame:
    if diag_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for family, group_df in diag_metrics.groupby("family_group"):
        rows.append(
            {
                "family_group": family,
                "n_codes": int(len(group_df)),
                "orig12_auroc_mean": float(np.nanmean(group_df["orig12_auroc"])),
                "recon12_auroc_mean": float(np.nanmean(group_df["recon12_auroc"])),
                "obs3_auroc_mean": float(np.nanmean(group_df["obs3_auroc"])),
                "recon_minus_obs3_mean": float(np.nanmean(group_df["recon_minus_obs3"])),
                "orig12_minus_recon_mean": float(np.nanmean(group_df["orig12_minus_recon"])),
                "recon_beats_obs3_rate": float(np.nanmean((group_df["recon12_auroc"] > group_df["obs3_auroc"]).astype(float))),
                "within_orig12_tolerance_rate": float(np.nanmean((group_df["recon12_auroc"] >= group_df["orig12_auroc"] - 0.05).astype(float))),
                "recovery_ratio_mean": float(np.nanmean(group_df["recovery_ratio"])),
            }
        )
    overall = pd.DataFrame(
        [
            {
                "family_group": "__overall__",
                "n_codes": int(len(diag_metrics)),
                "orig12_auroc_mean": float(np.nanmean(diag_metrics["orig12_auroc"])),
                "recon12_auroc_mean": float(np.nanmean(diag_metrics["recon12_auroc"])),
                "obs3_auroc_mean": float(np.nanmean(diag_metrics["obs3_auroc"])),
                "recon_minus_obs3_mean": float(np.nanmean(diag_metrics["recon_minus_obs3"])),
                "orig12_minus_recon_mean": float(np.nanmean(diag_metrics["orig12_minus_recon"])),
                "recon_beats_obs3_rate": float(np.nanmean((diag_metrics["recon12_auroc"] > diag_metrics["obs3_auroc"]).astype(float))),
                "within_orig12_tolerance_rate": float(np.nanmean((diag_metrics["recon12_auroc"] >= diag_metrics["orig12_auroc"] - 0.05).astype(float))),
                "recovery_ratio_mean": float(np.nanmean(diag_metrics["recovery_ratio"])),
            }
        ]
    )
    return pd.concat([pd.DataFrame(rows), overall], ignore_index=True)


def write_diag_code_interpretation(
    run_dir: Path,
    diag_catalog: pd.DataFrame,
    diag_metrics: pd.DataFrame,
    diag_family_summary: pd.DataFrame,
    singleton_audit: pd.DataFrame,
) -> None:
    supported = diag_catalog[diag_catalog["included_in_scored_metrics"]].copy()
    singleton = diag_catalog[diag_catalog["support_tier"] == "singleton_audit"].copy()
    overall = diag_family_summary[diag_family_summary["family_group"] == "__overall__"].iloc[0].to_dict() if not diag_family_summary.empty else {}
    top_scored = diag_metrics.sort_values("recon_minus_obs3", ascending=False).head(10) if not diag_metrics.empty else pd.DataFrame()
    lines = [
        "# Sunnybrook Diagnostic-Code Classification",
        "",
        "## Headline",
        f"- Literal diag_codes considered: `{int(len(diag_catalog))}`",
        f"- Scored literal diag_codes: `{int(len(supported))}`",
        f"- Singleton audit-only diag_codes: `{int(len(singleton))}`",
    ]
    if overall:
        lines.extend(
            [
                f"- Recon12 mean AUROC across scored codes: `{overall.get('recon12_auroc_mean', float('nan')):.3f}`",
                f"- Obs3 mean AUROC across scored codes: `{overall.get('obs3_auroc_mean', float('nan')):.3f}`",
                f"- Recon beats raw 3-lead across scored codes: `{overall.get('recon_beats_obs3_rate', float('nan')):.3f}`",
            ]
        )
    lines.extend(["", "## Scored Literal Codes"])
    for row in supported.itertuples(index=False):
        lines.append(f"- `{row.diag_code}`: positives=`{int(row.positive_count)}`, family=`{row.family_group}`")
    lines.extend(["", "## Strongest Recon-over-3L Literal-Code Gains"])
    for row in top_scored.itertuples(index=False):
        lines.append(
            f"- `{row.diag_code}`: recon AUROC `{row.recon12_auroc:.3f}`, obs3 AUROC `{row.obs3_auroc:.3f}`, "
            f"recon_minus_obs3 `{row.recon_minus_obs3:.3f}`"
        )
    lines.extend(["", "## Singleton Audit Codes"])
    if singleton_audit.empty:
        lines.append("- No singleton audit codes were available.")
    else:
        for row in singleton_audit.itertuples(index=False):
            lines.append(
                f"- `{row.diag_code}` on `{row.positive_file}`: "
                f"recon rank `{int(row.recon12_positive_rank)}`/20, "
                f"orig rank `{int(row.orig12_positive_rank)}`/20, "
                f"obs3 rank `{int(row.obs3_positive_rank)}`/20"
            )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- This report uses only literal `diag_codes` from Sunnybrook.",
            "- No text mining, grouped target construction, or clinical proxy expansion is used in the target definition.",
            "- Singleton codes are kept as audit entries only; they are not used in headline AUROC claims.",
        ]
    )
    (run_dir / "sunnybrook_diag_code_interpretation.md").write_text("\n".join(lines), encoding="ascii")


def build_signal_lookup(signal: np.ndarray, available_leads: list[int]) -> dict[str, np.ndarray]:
    return {LEAD_ORDER[idx]: signal[pos] for pos, idx in enumerate(available_leads)}


def extract_direct_features(signal_lookup: dict[str, np.ndarray], fs: int = 500) -> dict[str, float]:
    try:
        import neurokit2 as nk
    except ImportError:
        return {}

    if "II" not in signal_lookup:
        return {}

    features: dict[str, float] = {}
    lead_ii = np.asarray(signal_lookup["II"], dtype=np.float64)
    try:
        clean, info = nk.ecg_process(lead_ii, sampling_rate=fs)
    except Exception:
        return {}

    rate = clean.get("ECG_Rate")
    if rate is not None:
        rate_valid = pd.Series(rate).dropna()
        if not rate_valid.empty:
            features["heart_rate"] = float(rate_valid.median())

    rpeaks = info.get("ECG_R_Peaks", [])
    if len(rpeaks) >= 2:
        rr_ms = np.diff(rpeaks) / fs * 1000.0
        try:
            delineation = nk.ecg_delineate(clean["ECG_Clean"], rpeaks, sampling_rate=fs, method="dwt")
            if isinstance(delineation, tuple):
                delineation = delineation[1]
        except Exception:
            delineation = {}

        q_peaks = np.asarray(delineation.get("ECG_Q_Peaks", []), dtype=float)
        s_peaks = np.asarray(delineation.get("ECG_S_Peaks", []), dtype=float)
        if q_peaks.size and s_peaks.size:
            valid = ~np.isnan(q_peaks) & ~np.isnan(s_peaks)
            if valid.any():
                qrs = (s_peaks[valid] - q_peaks[valid]) / fs * 1000.0
                qrs = qrs[(qrs > 20) & (qrs < 300)]
                if qrs.size:
                    features["qrs_duration"] = float(np.median(qrs))

        p_onsets = np.asarray(delineation.get("ECG_P_Onsets", []), dtype=float)
        r_onsets = np.asarray(delineation.get("ECG_R_Onsets", []), dtype=float)
        if p_onsets.size and r_onsets.size:
            valid = ~np.isnan(p_onsets) & ~np.isnan(r_onsets)
            if valid.any():
                pr = (r_onsets[valid] - p_onsets[valid]) / fs * 1000.0
                pr = pr[(pr > 40) & (pr < 400)]
                if pr.size:
                    features["pr_interval"] = float(np.median(pr))

        t_offsets = np.asarray(delineation.get("ECG_T_Offsets", []), dtype=float)
        if r_onsets.size and t_offsets.size:
            valid = ~np.isnan(r_onsets) & ~np.isnan(t_offsets)
            if valid.any():
                qt = (t_offsets[valid] - r_onsets[valid]) / fs * 1000.0
                qt = qt[(qt > 150) & (qt < 700)]
                if qt.size:
                    qt_median = float(np.median(qt))
                    features["qt_interval"] = qt_median
                    rr_sec = float(np.median(rr_ms) / 1000.0) if len(rr_ms) else np.nan
                    if rr_sec and rr_sec > 0:
                        features["qtc"] = qt_median / math.sqrt(rr_sec)

    if "I" in signal_lookup and "aVF" in signal_lookup:
        lead_i = np.asarray(signal_lookup["I"], dtype=np.float64)
        lead_avf = np.asarray(signal_lookup["aVF"], dtype=np.float64)
        try:
            _, info_i = nk.ecg_process(lead_i, sampling_rate=fs)
            _, info_avf = nk.ecg_process(lead_avf, sampling_rate=fs)
            rp_i = info_i.get("ECG_R_Peaks", [])
            rp_avf = info_avf.get("ECG_R_Peaks", [])
            if len(rp_i) >= 2 and len(rp_avf) >= 2:
                net_i = np.mean([lead_i[max(0, p - 10) : p + 10].sum() for p in rp_i[:5] if p < len(lead_i) - 10])
                net_avf = np.mean([lead_avf[max(0, p - 10) : p + 10].sum() for p in rp_avf[:5] if p < len(lead_avf) - 10])
                features["qrs_axis"] = float(np.degrees(np.arctan2(net_avf, net_i)))
        except Exception:
            pass

    return features


def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    try:
        return float(func(y_true, y_pred)[0])
    except Exception:
        return float("nan")


def direct_feature_audit(
    files: list[str],
    arrays_by_source: dict[str, np.ndarray],
    obs_indices: list[int],
    target_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_record_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    source_leads = {
        "orig12": list(range(12)),
        "recon12": list(range(12)),
        "obs3": obs_indices,
    }

    for source_name, available in source_leads.items():
        for file_name, signal in zip(files, arrays_by_source[source_name]):
            lookup = build_signal_lookup(signal, available)
            extracted = extract_direct_features(lookup)
            for feature in DIRECT_FEATURES:
                target_val = pd.to_numeric(target_values.loc[file_name, feature], errors="coerce")
                pred_val = extracted.get(feature, np.nan)
                per_record_rows.append(
                    {
                        "file": file_name,
                        "source": source_name,
                        "feature": feature,
                        "target_value": float(target_val) if pd.notna(target_val) else np.nan,
                        "predicted_value": float(pred_val) if pred_val is not None and not math.isnan(pred_val) else np.nan,
                        "abs_error": float(abs(pred_val - target_val)) if pd.notna(target_val) and pred_val is not None and not math.isnan(pred_val) else np.nan,
                    }
                )

        direct_df = pd.DataFrame(per_record_rows)
        for feature in DIRECT_FEATURES:
            subset = direct_df[(direct_df["source"] == source_name) & (direct_df["feature"] == feature)].dropna(subset=["target_value", "predicted_value"])
            if subset.empty:
                metric_rows.append(
                    {
                        "source": source_name,
                        "feature": feature,
                        "n": 0,
                        "pearson": np.nan,
                        "spearman": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                    }
                )
                continue
            y_true = subset["target_value"].to_numpy(dtype=float)
            y_pred = subset["predicted_value"].to_numpy(dtype=float)
            metric_rows.append(
                {
                    "source": source_name,
                    "feature": feature,
                    "n": int(len(subset)),
                    "pearson": safe_corr(pearsonr, y_true, y_pred),
                    "spearman": safe_corr(spearmanr, y_true, y_pred),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(per_record_rows)


def flatten_sources(arrays_by_source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "orig12": arrays_by_source["orig12"].reshape(arrays_by_source["orig12"].shape[0], -1),
        "recon12": arrays_by_source["recon12"].reshape(arrays_by_source["recon12"].shape[0], -1),
        "obs3": arrays_by_source["obs3"].reshape(arrays_by_source["obs3"].shape[0], -1),
    }


def build_fold_embeddings(flattened: dict[str, np.ndarray], pca_dim: int) -> dict[str, dict[int, dict[str, Any]]]:
    n = next(iter(flattened.values())).shape[0]
    cache: dict[str, dict[int, dict[str, Any]]] = {}
    for source_name, matrix in flattened.items():
        cache[source_name] = {}
        for test_idx in range(n):
            train_idx = np.array([i for i in range(n) if i != test_idx], dtype=int)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(matrix[train_idx])
            x_test = scaler.transform(matrix[[test_idx]])
            k = max(1, min(pca_dim, x_train.shape[0] - 2, x_train.shape[1]))
            if k < x_train.shape[1]:
                pca = PCA(n_components=k, svd_solver="auto", random_state=0)
                x_train = pca.fit_transform(x_train)
                x_test = pca.transform(x_test)
            cache[source_name][test_idx] = {
                "train_idx": train_idx,
                "x_train": x_train,
                "x_test": x_test[0],
            }
    return cache


def collect_regression_predictions(
    target_name: str,
    y: pd.Series,
    cache: dict[str, dict[int, dict[str, Any]]],
    files: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    for source_name, source_cache in cache.items():
        preds = np.full_like(y_values, np.nan, dtype=float)
        for test_idx, fold in source_cache.items():
            if math.isnan(y_values[test_idx]):
                continue
            train_idx = fold["train_idx"]
            valid_train = ~np.isnan(y_values[train_idx])
            if valid_train.sum() < 3:
                continue
            x_train = fold["x_train"][valid_train]
            y_train = y_values[train_idx][valid_train]
            if np.nanstd(y_train) < 1e-8:
                continue
            model = Ridge(alpha=1.0)
            model.fit(x_train, y_train)
            preds[test_idx] = float(model.predict(fold["x_test"][None])[0])

        valid = ~np.isnan(y_values) & ~np.isnan(preds)
        for idx in np.where(valid)[0]:
            rows.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "file": files[idx],
                    "y_true": y_values[idx],
                    "y_pred": preds[idx],
                }
            )
        if valid.sum() < 3:
            metric_rows.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "n": int(valid.sum()),
                    "spearman": np.nan,
                    "pearson": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                }
            )
            continue
        truth = y_values[valid]
        pred = preds[valid]
        metric_rows.append(
            {
                "target": target_name,
                "source": source_name,
                "n": int(valid.sum()),
                "spearman": safe_corr(spearmanr, truth, pred),
                "pearson": safe_corr(pearsonr, truth, pred),
                "mae": float(mean_absolute_error(truth, pred)),
                "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
                "r2": float(r2_score(truth, pred)),
            }
        )
    return rows, metric_rows


def collect_classification_predictions(
    target_name: str,
    y: pd.Series,
    cache: dict[str, dict[int, dict[str, Any]]],
    files: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    for source_name, source_cache in cache.items():
        preds = np.full_like(y_values, np.nan, dtype=float)
        for test_idx, fold in source_cache.items():
            if math.isnan(y_values[test_idx]):
                continue
            train_idx = fold["train_idx"]
            valid_train = ~np.isnan(y_values[train_idx])
            x_train = fold["x_train"][valid_train]
            y_train = y_values[train_idx][valid_train].astype(int)
            if len(np.unique(y_train)) < 2:
                continue
            model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=0)
            model.fit(x_train, y_train)
            preds[test_idx] = float(model.predict_proba(fold["x_test"][None])[0, 1])

        valid = ~np.isnan(y_values) & ~np.isnan(preds)
        for idx in np.where(valid)[0]:
            rows.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "file": files[idx],
                    "y_true": int(y_values[idx]),
                    "y_score": preds[idx],
                    "y_pred": int(preds[idx] >= 0.5),
                }
            )
        valid_truth = y_values[valid].astype(int)
        valid_score = preds[valid]
        if valid.sum() < 3 or len(np.unique(valid_truth)) < 2:
            metric_rows.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "n": int(valid.sum()),
                    "prevalence": float(valid_truth.mean()) if valid.sum() else np.nan,
                    "auroc": np.nan,
                    "average_precision": np.nan,
                    "balanced_accuracy": np.nan,
                }
            )
            continue
        metric_rows.append(
            {
                "target": target_name,
                "source": source_name,
                "n": int(valid.sum()),
                "prevalence": float(valid_truth.mean()),
                "auroc": float(roc_auc_score(valid_truth, valid_score)),
                "average_precision": float(average_precision_score(valid_truth, valid_score)),
                "balanced_accuracy": float(balanced_accuracy_score(valid_truth, (valid_score >= 0.5).astype(int))),
            }
        )
    return rows, metric_rows


def pivot_result_table(long_df: pd.DataFrame, id_columns: list[str], metric_columns: list[str], primary_metric: str) -> pd.DataFrame:
    pieces = []
    for source_name in ["orig12", "recon12", "obs3"]:
        subset = long_df[long_df["source"] == source_name].copy()
        rename = {metric: f"{source_name}_{metric}" for metric in metric_columns}
        subset = subset[id_columns + metric_columns].rename(columns=rename)
        pieces.append(subset)
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.merge(piece, on=id_columns, how="outer")
    wide["recon_minus_obs3"] = wide[f"recon12_{primary_metric}"] - wide[f"obs3_{primary_metric}"]
    wide["orig12_minus_recon"] = wide[f"orig12_{primary_metric}"] - wide[f"recon12_{primary_metric}"]
    denom = wide[f"orig12_{primary_metric}"] - wide[f"obs3_{primary_metric}"]
    numer = wide[f"recon12_{primary_metric}"] - wide[f"obs3_{primary_metric}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        recovery = numer / denom
    wide["recovery_ratio"] = np.clip(recovery, -2.0, 2.0)
    return wide


def build_grouped_summary(
    regression_results: pd.DataFrame,
    classification_results: pd.DataFrame,
    catalog: pd.DataFrame,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    group_map = catalog.set_index("canonical_name")[["group", "support_tier"]]
    rows: list[dict[str, Any]] = []
    for result_df, target_type, primary_metric in [
        (regression_results, "regression", "spearman"),
        (classification_results, "classification", "auroc"),
    ]:
        if result_df.empty or "target" not in result_df.columns:
            continue
        merged = result_df.merge(group_map, left_on="target", right_index=True, how="left")
        if "support_tier" not in merged.columns:
            left_tier = merged["support_tier_x"] if "support_tier_x" in merged.columns else None
            right_tier = merged["support_tier_y"] if "support_tier_y" in merged.columns else None
            if left_tier is not None and right_tier is not None:
                merged["support_tier"] = left_tier.combine_first(right_tier)
            elif left_tier is not None:
                merged["support_tier"] = left_tier
            else:
                merged["support_tier"] = right_tier
        if "group" not in merged.columns:
            left_group = merged["group_x"] if "group_x" in merged.columns else None
            right_group = merged["group_y"] if "group_y" in merged.columns else None
            if left_group is not None and right_group is not None:
                merged["group"] = left_group.combine_first(right_group)
            elif left_group is not None:
                merged["group"] = left_group
            else:
                merged["group"] = right_group
        merged = merged[merged["support_tier"].isin(["supported", "exploratory"])]
        for group_name, group_df in merged.groupby("group"):
            rows.append(
                summarize_group(group_df, group_name, target_type, primary_metric, tolerance)
            )
        rows.append(summarize_group(merged, "__overall__", target_type, primary_metric, tolerance))
    return pd.DataFrame(rows)


def summarize_group(
    df: pd.DataFrame,
    group_name: str,
    target_type: str,
    primary_metric: str,
    tolerance: float,
) -> dict[str, Any]:
    recon = df[f"recon12_{primary_metric}"].to_numpy(dtype=float)
    obs = df[f"obs3_{primary_metric}"].to_numpy(dtype=float)
    orig = df[f"orig12_{primary_metric}"].to_numpy(dtype=float)
    valid = ~np.isnan(recon) & ~np.isnan(obs) & ~np.isnan(orig)
    if not valid.any():
        return {
            "target_group": group_name,
            "target_type": target_type,
            "n_targets": 0,
            "orig12_primary_mean": np.nan,
            "recon12_primary_mean": np.nan,
            "obs3_primary_mean": np.nan,
            "recon_minus_obs3_mean": np.nan,
            "orig12_minus_recon_mean": np.nan,
            "recon_beats_obs3_rate": np.nan,
            "within_orig12_tolerance_rate": np.nan,
            "recovery_ratio_mean": np.nan,
        }
    subset = df.loc[valid]
    return {
        "target_group": group_name,
        "target_type": target_type,
        "n_targets": int(len(subset)),
        "orig12_primary_mean": float(np.nanmean(subset[f"orig12_{primary_metric}"])),
        "recon12_primary_mean": float(np.nanmean(subset[f"recon12_{primary_metric}"])),
        "obs3_primary_mean": float(np.nanmean(subset[f"obs3_{primary_metric}"])),
        "recon_minus_obs3_mean": float(np.nanmean(subset["recon_minus_obs3"])),
        "orig12_minus_recon_mean": float(np.nanmean(subset["orig12_minus_recon"])),
        "recon_beats_obs3_rate": float(np.nanmean((subset[f"recon12_{primary_metric}"] > subset[f"obs3_{primary_metric}"]).astype(float))),
        "within_orig12_tolerance_rate": float(np.nanmean((subset[f"recon12_{primary_metric}"] >= subset[f"orig12_{primary_metric}"] - tolerance).astype(float))),
        "recovery_ratio_mean": float(np.nanmean(subset["recovery_ratio"])),
    }


def write_interpretation(
    run_dir: Path,
    regression_results: pd.DataFrame,
    classification_results: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    direct_metrics: pd.DataFrame,
    catalog: pd.DataFrame,
) -> None:
    overall_reg = grouped_summary[(grouped_summary["target_group"] == "__overall__") & (grouped_summary["target_type"] == "regression")]
    overall_cls = grouped_summary[(grouped_summary["target_group"] == "__overall__") & (grouped_summary["target_type"] == "classification")]
    reg_row = overall_reg.iloc[0].to_dict() if not overall_reg.empty else {}
    cls_row = overall_cls.iloc[0].to_dict() if not overall_cls.empty else {}

    top_reg = regression_results.sort_values("recon_minus_obs3", ascending=False).head(5)
    top_cls = classification_results.sort_values("recon_minus_obs3", ascending=False).head(5)
    supported_counts = catalog["support_tier"].value_counts(dropna=False).to_dict()

    lines = [
        "# Sunnybrook All-Feature Baseline VAE Benchmark",
        "",
        "## Headline",
        f"- Supported/exploratory regression targets: `{int(reg_row.get('n_targets', 0))}`",
        f"- Supported/exploratory classification targets: `{int(cls_row.get('n_targets', 0))}`",
        f"- Recon beats raw 3-lead on regression targets: `{reg_row.get('recon_beats_obs3_rate', float('nan')):.3f}`",
        f"- Recon beats raw 3-lead on classification targets: `{cls_row.get('recon_beats_obs3_rate', float('nan')):.3f}`",
        f"- Recon within tolerance of original 12-lead on regression targets: `{reg_row.get('within_orig12_tolerance_rate', float('nan')):.3f}`",
        f"- Recon within tolerance of original 12-lead on classification targets: `{cls_row.get('within_orig12_tolerance_rate', float('nan')):.3f}`",
        "",
        "## Direct Zero-Shot Feature Preservation",
    ]

    for feature in DIRECT_FEATURES:
        subset = direct_metrics[direct_metrics["feature"] == feature]
        if subset.empty:
            continue
        parts = [f"`{row.source}` spearman={row.spearman:.3f} mae={row.mae:.3f}" for row in subset.itertuples()]
        lines.append(f"- `{feature}`: " + "; ".join(parts))

    lines.extend(["", "## Strongest Recon-over-3L Regression Gains"])
    for row in top_reg.itertuples():
        lines.append(f"- `{row.target}`: recon_minus_obs3={row.recon_minus_obs3:.3f}, orig12_minus_recon={row.orig12_minus_recon:.3f}")

    lines.extend(["", "## Strongest Recon-over-3L Classification Gains"])
    for row in top_cls.itertuples():
        lines.append(f"- `{row.target}`: recon_minus_obs3={row.recon_minus_obs3:.3f}, orig12_minus_recon={row.orig12_minus_recon:.3f}")

    lines.extend(
        [
            "",
            "## Support Tiers",
            f"- counts: `{json.dumps(supported_counts, sort_keys=True)}`",
            "",
            "## Interpretation Notes",
            "- Original 12-lead scores are a ceiling/reference, not an external gold-standard classifier.",
            "- Reconstruction results are descriptive only because Sunnybrook contains 20 ECGs.",
            "- Unsupported and highly imbalanced labels remain in the catalog for auditability, but are not interpreted as reliable endpoints.",
        ]
    )
    (run_dir / "interpretation.md").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_indices = parse_obs_leads(args.obs_leads)
    obs_leads = [LEAD_ORDER[idx] for idx in obs_indices]

    model, ckpt, resolved_family = load_reconstruction_model(args.checkpoint, device, args.model_family)

    all_xml = sorted(Path(args.sunnybrook_dir).glob("*.xml"))
    if args.limit is not None:
        all_xml = all_xml[: args.limit]
    files = [path.name for path in all_xml]

    checkpoint_path = Path(args.checkpoint).resolve()
    run_name = f"{checkpoint_path.parent.name}__{checkpoint_path.stem}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        str(run_dir / "run_metadata.json"),
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "run_name": run_name,
            "model_family": resolved_family,
            "obs_leads": obs_leads,
            "obs_lead_indices": obs_indices,
            "sunnybrook_dir": str(Path(args.sunnybrook_dir).resolve()),
            "master_csv": str(Path(args.master_csv).resolve()),
            "features_csv": str(Path(args.features_csv).resolve()),
            "extra_csv": str(Path(args.extra_csv).resolve()),
            "device": str(device),
            "pca_dim": int(args.pca_dim),
            "limit": args.limit,
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        },
    )

    manifest_df, arrays_by_source = reconstruct_sources(model, all_xml, obs_indices, device, args.use_amp)
    manifest_df.to_csv(run_dir / "waveform_sources_manifest.csv", index=False)

    catalog_df, target_values = make_target_catalog(args.master_csv, args.features_csv, args.extra_csv)
    target_values = target_values.reindex(files)
    catalog_df.to_csv(run_dir / "target_catalog.csv", index=False)
    diag_code_catalog = build_diag_code_catalog(target_values)
    diag_code_catalog.to_csv(run_dir / "sunnybrook_diag_code_catalog.csv", index=False)

    direct_metrics_df, direct_record_df = direct_feature_audit(files, arrays_by_source, obs_indices, target_values)
    direct_metrics_df.to_csv(run_dir / "direct_feature_metrics.csv", index=False)
    direct_record_df.to_csv(run_dir / "direct_feature_record_errors.csv", index=False)

    flattened = flatten_sources(arrays_by_source)
    fold_cache = build_fold_embeddings(flattened, args.pca_dim)

    regression_pred_rows: list[dict[str, Any]] = []
    regression_metric_rows: list[dict[str, Any]] = []
    classification_pred_rows: list[dict[str, Any]] = []
    classification_metric_rows: list[dict[str, Any]] = []

    supported_catalog = catalog_df[catalog_df["support_tier"].isin(["supported", "exploratory"])].copy()
    for row in tqdm(list(supported_catalog.itertuples(index=False)), desc="Running LOOCV probes"):
        target_name = row.canonical_name
        y = target_values[target_name]
        if row.target_type == "regression":
            pred_rows, metric_rows = collect_regression_predictions(target_name, y, fold_cache, files)
            regression_pred_rows.extend(pred_rows)
            regression_metric_rows.extend(metric_rows)
        elif row.target_type == "classification":
            pred_rows, metric_rows = collect_classification_predictions(target_name, y, fold_cache, files)
            classification_pred_rows.extend(pred_rows)
            classification_metric_rows.extend(metric_rows)

    regression_long = pd.DataFrame(regression_metric_rows)
    classification_long = pd.DataFrame(classification_metric_rows)
    regression_preds = pd.DataFrame(regression_pred_rows)
    classification_preds = pd.DataFrame(classification_pred_rows)

    if not regression_preds.empty:
        regression_preds.to_csv(run_dir / "all_feature_regression_predictions.csv", index=False)
    if not classification_preds.empty:
        classification_preds.to_csv(run_dir / "all_feature_classification_predictions.csv", index=False)

    regression_results = pivot_result_table(
        regression_long,
        ["target"],
        ["n", "spearman", "pearson", "mae", "rmse", "r2"],
        "spearman",
    ) if not regression_long.empty else pd.DataFrame()
    classification_results = pivot_result_table(
        classification_long,
        ["target"],
        ["n", "prevalence", "auroc", "average_precision", "balanced_accuracy"],
        "auroc",
    ) if not classification_long.empty else pd.DataFrame()

    if not regression_results.empty:
        regression_results = regression_results.merge(
            catalog_df[["canonical_name", "group", "support_tier"]].rename(columns={"canonical_name": "target"}),
            on="target",
            how="left",
        )
        regression_results.to_csv(run_dir / "all_feature_regression_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "all_feature_regression_results.csv", index=False)

    if not classification_results.empty:
        classification_results = classification_results.merge(
            catalog_df[["canonical_name", "group", "support_tier"]].rename(columns={"canonical_name": "target"}),
            on="target",
            how="left",
        )
        classification_results.to_csv(run_dir / "all_feature_classification_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "all_feature_classification_results.csv", index=False)

    grouped_summary = build_grouped_summary(
        regression_results if not regression_results.empty else pd.DataFrame(columns=["target"]),
        classification_results if not classification_results.empty else pd.DataFrame(columns=["target"]),
        catalog_df,
    )
    grouped_summary.to_csv(run_dir / "grouped_summary.csv", index=False)

    diag_code_metrics = build_diag_code_metrics(
        classification_results if not classification_results.empty else pd.DataFrame(),
        diag_code_catalog,
    )
    diag_code_metrics.to_csv(run_dir / "sunnybrook_diag_code_metrics.csv", index=False)
    diag_code_family_summary = summarize_diag_code_families(diag_code_metrics)
    diag_code_family_summary.to_csv(run_dir / "sunnybrook_diag_code_family_summary.csv", index=False)
    diag_code_singleton_audit = build_full_cohort_diag_code_audit(
        files,
        flattened,
        target_values,
        diag_code_catalog,
        args.pca_dim,
    )
    diag_code_singleton_audit.to_csv(run_dir / "sunnybrook_diag_code_singleton_audit.csv", index=False)
    write_diag_code_interpretation(
        run_dir,
        diag_code_catalog,
        diag_code_metrics,
        diag_code_family_summary,
        diag_code_singleton_audit,
    )

    write_interpretation(
        run_dir,
        regression_results if not regression_results.empty else pd.DataFrame(),
        classification_results if not classification_results.empty else pd.DataFrame(),
        grouped_summary,
        direct_metrics_df,
        catalog_df,
    )

    print(f"Saved Sunnybrook all-feature benchmark to {run_dir}")


if __name__ == "__main__":
    main()
