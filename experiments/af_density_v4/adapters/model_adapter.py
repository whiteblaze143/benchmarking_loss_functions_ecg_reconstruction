"""Mandatory Model Adapter Interfaces (V4).

Defines the universal adapter contract for existing detection models:
- RR-GBDT (Non-neural champion)
- ArNet2 (Autoregressive beat sequence)
- RawECGNet (1D Raw voltage CNN)
- H071-v2 (Per-channel raw ResNet)
- G061 (Multimodal RR + Phase-KME SetOperator)
- A001 (Canonical continuous SetOperator)
- Oracle Majority (Theoretical discretization upper bound)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Sequence
import time
import numpy as np
import torch


@dataclass(frozen=True)
class WindowPrediction:
    """Standardized output record for an individual 30-second observation window."""
    dataset_id: str
    subject_id: str
    record_id: str
    source_id: str
    start_s: float
    end_s: float
    raw_af_score: float
    score_interpretation: str  # 'logit', 'probability', 'margin'
    calibrated_af_prob: Optional[float]
    lead_availability: Tuple[bool, bool]
    acquisition_configuration: str  # 'Pattern 0', 'Pattern 1', 'Pattern 2'
    signal_quality_valid: bool
    rr_extraction_valid: bool
    prediction_status: str  # 'valid', 'abstained', 'technical_failure'
    model_id: str
    checkpoint_hash: str
    processing_time_ms: float
    resampling_applied: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lead_availability"] = list(self.lead_availability)
        return d


class AFModelAdapter(ABC):
    """Abstract Base Class for all V4 AF detection model adapters."""

    def __init__(self, model_id: str, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.checkpoint_hash = self._compute_checkpoint_hash()

    def _compute_checkpoint_hash(self) -> str:
        if not self.checkpoint_path:
            return "none_prespecified"
        import hashlib
        from pathlib import Path
        p = Path(self.checkpoint_path)
        if not p.exists():
            return "checkpoint_not_found"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    @abstractmethod
    def predict(
        self,
        record: Dict[str, Any],
        observation_mask: Optional[np.ndarray] = None,
        configuration: str = "Pattern 2 (Both leads)"
    ) -> List[WindowPrediction]:
        """Performs causal/offline inference across the recording.
        
        Args:
            record: Dict containing:
                - 'dataset_id': str
                - 'subject_id': str
                - 'record_id': str
                - 'raw_waveforms': Optional (N, 2, 6000) float32 array
                - 'rr_features': Optional (N, 36) float32 array
                - 'rr_valid': Optional (N, 2) bool array
                - 'phase_features': Optional (N, 2, 16, 128) float32 array
                - 'window_timestamps': (N, 2) array of [start_s, end_s]
            observation_mask: Optional bool array (N, 2) indicating unobservable leads
            configuration: acquisition pattern name
            
        Returns:
            List of WindowPrediction objects.
        """
        pass


class RRGBDTAdapter(AFModelAdapter):
    """Adapter for Non-Neural RR-GBDT (HistGradientBoostingClassifier)."""

    def __init__(self, model_instance, checkpoint_hash: str = "rr_gbdt_canonical"):
        super().__init__(model_id="RR-GBDT", device="cpu")
        self.clf = model_instance
        self.checkpoint_hash = checkpoint_hash

    def predict(
        self,
        record: Dict[str, Any],
        observation_mask: Optional[np.ndarray] = None,
        configuration: str = "Pattern 2 (Both leads)"
    ) -> List[WindowPrediction]:
        t0 = time.perf_counter()
        dataset_id = record.get("dataset_id", "IRIDIA-AF")
        subject_id = record.get("subject_id", "unknown_subject")
        record_id = record.get("record_id", "unknown_record")
        rr = record["rr_features"]
        timestamps = record["window_timestamps"]
        N = len(rr)

        # Enforce genuine lead-dropout in RR features
        x = rr.copy()
        p0 = "Pattern 0" in configuration or "Lead 0" in configuration
        p1 = "Pattern 1" in configuration or "Lead 1" in configuration
        
        lead_avail = (True, True)
        if p0 and not p1:
            x[:, 16:32] = 0.0  # Zero Lead 1 features
            x[:, 33] = 0.0     # mask_ch1 = 0
            x[:, 35] = 0.0     # valid_ch1 = 0
            lead_avail = (True, False)
        elif p1 and not p0:
            x[:, 0:16] = 0.0   # Zero Lead 0 features
            x[:, 32] = 0.0     # mask_ch0 = 0
            x[:, 34] = 0.0     # valid_ch0 = 0
            lead_avail = (False, True)

        x = np.nan_to_num(x, nan=0.0)
        probs = self.clf.predict_proba(x)[:, 1]
        elapsed_per_win = ((time.perf_counter() - t0) * 1000.0) / max(1, N)

        preds = []
        for i in range(N):
            p = float(probs[i])
            preds.append(
                WindowPrediction(
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    record_id=record_id,
                    source_id=f"{record_id}_win_{i:06d}",
                    start_s=float(timestamps[i, 0]),
                    end_s=float(timestamps[i, 1]),
                    raw_af_score=p,
                    score_interpretation="probability",
                    calibrated_af_prob=p,
                    lead_availability=lead_avail,
                    acquisition_configuration=configuration,
                    signal_quality_valid=True,
                    rr_extraction_valid=bool(x[i, 34] > 0 or x[i, 35] > 0),
                    prediction_status="valid",
                    model_id="RR-GBDT",
                    checkpoint_hash=self.checkpoint_hash,
                    processing_time_ms=elapsed_per_win
                )
            )
        return preds


class NeuralModelAdapter(AFModelAdapter):
    """Adapter for PyTorch Deep Learning Models (ArNet2, RawECGNet, H071-v2, G061, A001)."""

    def __init__(self, model_id: str, model_instance: torch.nn.Module, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_id=model_id, checkpoint_path=checkpoint_path, device=device)
        self.model = model_instance.to(device)
        self.model.eval()

    def predict(
        self,
        record: Dict[str, Any],
        observation_mask: Optional[np.ndarray] = None,
        configuration: str = "Pattern 2 (Both leads)"
    ) -> List[WindowPrediction]:
        t0 = time.perf_counter()
        dataset_id = record.get("dataset_id", "IRIDIA-AF")
        subject_id = record.get("subject_id", "unknown_subject")
        record_id = record.get("record_id", "unknown_record")
        timestamps = record["window_timestamps"]
        N = len(timestamps)

        raw = record.get("raw_waveforms")
        resp = record.get("phase_features")
        rr = record.get("rr_features")

        lead_avail = (True, True)
        c_mask = torch.zeros(N, 2, dtype=torch.bool, device=self.device)
        ops = torch.zeros(N, 2, 8, device=self.device)

        if "Pattern 0" in configuration:
            c_mask[:, 1] = True
            ops[:, 0, 0] = 1.0
            lead_avail = (True, False)
            if raw is not None:
                raw[:, 1, :] = 0.0
            if rr is not None:
                rr[:, 16:32] = 0.0
                rr[:, 33] = 0.0
                rr[:, 35] = 0.0
        elif "Pattern 1" in configuration:
            c_mask[:, 0] = True
            ops[:, 1, 1] = 1.0
            lead_avail = (False, True)
            if raw is not None:
                raw[:, 0, :] = 0.0
            if rr is not None:
                rr[:, 0:16] = 0.0
                rr[:, 32] = 0.0
                rr[:, 34] = 0.0
        else:
            ops[:, 0, 0] = 1.0
            ops[:, 1, 1] = 1.0

        all_logits = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, N, batch_size):
                end_idx = min(N, i + batch_size)
                b_cmask = c_mask[i:end_idx]
                b_ops = ops[i:end_idx]

                if self.model_id in ["H071-v2", "RawECGNet"]:
                    sub_raw = torch.from_numpy(raw[i:end_idx]).to(self.device).float()
                    out = self.model(sub_raw, b_ops, b_cmask)
                elif self.model_id in ["G061"]:
                    sub_resp = torch.from_numpy(resp[i:end_idx]).to(self.device).float()
                    sub_rr = torch.from_numpy(rr[i:end_idx]).to(self.device).float()
                    out = self.model(sub_resp, b_ops, b_cmask, sub_rr)
                elif self.model_id in ["ArNet2"]:
                    # Autoregressive beat sequence model
                    sub_rr = torch.from_numpy(rr[i:end_idx, :16]).to(self.device).float()
                    out = self.model(sub_rr)
                else:  # A001 / Continuous SetOperator
                    sub_resp = torch.from_numpy(resp[i:end_idx]).to(self.device).float()
                    out = self.model(sub_resp, b_ops, b_cmask)

                if isinstance(out, dict):
                    logits = out["logits"].squeeze(-1)
                else:
                    logits = out.squeeze(-1)
                all_logits.append(logits.cpu().numpy())

        logits = np.concatenate(all_logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        elapsed_per_win = ((time.perf_counter() - t0) * 1000.0) / max(1, N)

        preds = []
        for i in range(N):
            preds.append(
                WindowPrediction(
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    record_id=record_id,
                    source_id=f"{record_id}_win_{i:06d}",
                    start_s=float(timestamps[i, 0]),
                    end_s=float(timestamps[i, 1]),
                    raw_af_score=float(logits[i]),
                    score_interpretation="logit",
                    calibrated_af_prob=float(probs[i]),
                    lead_availability=lead_avail,
                    acquisition_configuration=configuration,
                    signal_quality_valid=True,
                    rr_extraction_valid=True,
                    prediction_status="valid",
                    model_id=self.model_id,
                    checkpoint_hash=self.checkpoint_hash,
                    processing_time_ms=elapsed_per_win
                )
            )
        return preds
