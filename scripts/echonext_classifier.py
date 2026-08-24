"""Frozen official EchoNext Mini-Model integration.

The implementation loads the public Columbia Mini-Model weights directly
without depending on PyTorch Lightning. Tabular preprocessing is reproduced
from the fitted official sklearn objects using their learned parameters; this
avoids incompatibility between the original sklearn 1.1.3 pickle and newer
runtime versions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly


SHD_TASKS = [
    "lvef_lte_45",
    "lvwt_gte_13",
    "aortic_stenosis_moderate_or_greater",
    "aortic_regurgitation_moderate_or_greater",
    "mitral_regurgitation_moderate_or_greater",
    "tricuspid_regurgitation_moderate_or_greater",
    "pulmonary_regurgitation_moderate_or_greater",
    "rv_systolic_dysfunction_moderate_or_greater",
    "pericardial_effusion_moderate_large",
    "pasp_gte_45",
    "tr_max_gte_32",
    "shd_moderate_or_greater",
]

SHD_LABEL_COLUMNS = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]

TABULAR_FLOAT_COLUMNS = [
    "age_at_ecg",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_echonext_test_metadata(
    metadata_path: Path,
    transformer_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(metadata_path)
    missing = sorted(
        {"split", "sex", *TABULAR_FLOAT_COLUMNS, *SHD_LABEL_COLUMNS} - set(metadata.columns)
    )
    if missing:
        raise ValueError(f"EchoNext metadata missing required columns: {missing}")
    test = metadata.loc[metadata["split"] == "test"].reset_index(drop=True)
    if len(test) != 5442:
        raise ValueError(f"Expected 5,442 EchoNext test records, found {len(test)}")

    clean = test.copy()
    clean["atrial_rate"] = clean["atrial_rate"].fillna(0)
    clean["pr_interval"] = clean["pr_interval"].fillna(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(transformer_path)
    try:
        scaler = pipeline.__dict__["steps"][0][1]
        imputer = pipeline.__dict__["steps"][1][1]
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        imputation = np.asarray(imputer.statistics_, dtype=np.float64)
    except (AttributeError, KeyError, IndexError) as error:
        raise ValueError("Unsupported official EchoNext tabular transformer") from error
    raw = clean[TABULAR_FLOAT_COLUMNS].to_numpy(dtype=np.float64)
    scaled = (raw - mean) / scale
    missing_values = np.where(np.isnan(scaled))
    scaled[missing_values] = np.take(imputation, missing_values[1])
    sex = (
        clean["sex"]
        .astype(str)
        .str.lower()
        .map({"female": 0.0, "male": 1.0})
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(sex).all():
        raise ValueError("EchoNext test metadata contains unsupported/missing sex values")
    tabular = np.concatenate([sex[:, None], scaled], axis=1).astype(np.float32)
    labels = test[SHD_LABEL_COLUMNS].to_numpy(dtype=np.float32)
    if tabular.shape != (5442, 7) or labels.shape != (5442, 12):
        raise ValueError(
            f"Unexpected EchoNext tabular/label shapes: {tabular.shape}, {labels.shape}"
        )
    if not np.isfinite(tabular).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("EchoNext tabular features or labels failed finite/binary validation")
    return test, tabular, labels


class EchoNextMiniModel:
    """Thin inference wrapper around the official frozen 12-task model."""

    def __init__(
        self,
        repository_root: Path,
        device: torch.device,
    ):
        self.repository_root = repository_root.resolve()
        model_root = self.repository_root / "models/echonext_multilabel_minimodel"
        self.weights_path = model_root / "weights.pt"
        self.transformer_path = model_root / "tabular_transformer.joblib"
        self.waveform_params_path = model_root / "waveform_normalization_params.json"
        architecture_path = self.repository_root / "cradlenet/models/resnet1d_tabular.py"
        for path in (
            self.weights_path,
            self.transformer_path,
            self.waveform_params_path,
            architecture_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing official EchoNext asset: {path}")

        spec = importlib.util.spec_from_file_location(
            "official_echonext_resnet1d_tabular",
            architecture_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import official EchoNext model: {architecture_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ResNet1dWithTabular(
            len_tabular_feature_vector=7,
            filter_size=16,
            num_classes=12,
        )
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        self.device = device
        self.model = model.to(device).eval()
        self.waveform_normalization = json.loads(self.waveform_params_path.read_text())

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "name": "EchoNext_Mini_multilabel",
            "source_repository": "https://github.com/PierreElias/IntroECG",
            "source_commit": "15233e9392e9dc10c136a5624abf10199207bce9",
            "repository_root": str(self.repository_root),
            "weights": str(self.weights_path),
            "weights_sha256": sha256_file(self.weights_path),
            "tabular_transformer": str(self.transformer_path),
            "tabular_transformer_sha256": sha256_file(self.transformer_path),
            "waveform_normalization_params": str(self.waveform_params_path),
            "waveform_normalization_params_sha256": sha256_file(
                self.waveform_params_path
            ),
            "tasks": SHD_TASKS,
            "threshold": 0.5,
            "threshold_source": "fixed_predefined_0.5",
        }

    @torch.inference_mode()
    def predict_official_waveforms(
        self,
        waveforms: np.ndarray | torch.Tensor,
        tabular: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Predict from official normalized [B,1,2500,12] waveforms."""
        waveform_tensor = torch.as_tensor(waveforms, dtype=torch.float32, device=self.device)
        tabular_tensor = torch.as_tensor(tabular, dtype=torch.float32, device=self.device)
        if waveform_tensor.ndim != 4 or waveform_tensor.shape[1:] != (1, 2500, 12):
            raise ValueError(
                f"Official EchoNext input must be [B,1,2500,12], got {waveform_tensor.shape}"
            )
        if tabular_tensor.shape != (waveform_tensor.shape[0], 7):
            raise ValueError(
                f"EchoNext tabular input must be [B,7], got {tabular_tensor.shape}"
            )
        return torch.sigmoid(self.model((waveform_tensor, tabular_tensor))).cpu().numpy()

    def predict_reconstruction_500hz(
        self,
        reconstruction: torch.Tensor,
        tabular: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Predict from physical-mV reconstruction output shaped [B,12,5000]."""
        if reconstruction.ndim != 3 or reconstruction.shape[1:] != (12, 5000):
            raise ValueError(
                f"Reconstruction must be [B,12,5000], got {reconstruction.shape}"
            )
        values_mv = reconstruction.detach().float().cpu().numpy()
        downsampled_uv = (
            resample_poly(values_mv, up=1, down=2, axis=-1).astype(np.float32)
            * 1000.0
        )
        params = self.waveform_normalization
        lower = np.asarray(params["lowerbound"], dtype=np.float32).reshape(1, 12, 1)
        upper = np.asarray(params["upperbound"], dtype=np.float32).reshape(1, 12, 1)
        mean = np.asarray(params["mean"], dtype=np.float32).reshape(1, 12, 1)
        std = np.asarray(params["std"], dtype=np.float32).reshape(1, 12, 1)
        standardized = (np.clip(downsampled_uv, lower, upper) - mean) / std
        official_layout = standardized.transpose(0, 2, 1)[:, None, :, :]
        return self.predict_official_waveforms(official_layout, tabular)
