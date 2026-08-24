import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.benchmark_factorial_inference_readiness import artifact_is_current


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/factorial_mixed_level/inference_readiness"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inference_readiness_artifact_is_generation_bound_and_complete():
    summary = json.loads((RESULT_ROOT / "summary.json").read_text())
    audit_path = ROOT / "results/checkpoint_store/compatibility_audit.json"
    code_path = ROOT / "scripts/benchmark_factorial_inference_readiness.py"
    csv_path = RESULT_ROOT / "per_model_inference_readiness.csv"
    lead_csv_path = RESULT_ROOT / "per_model_per_lead_case_metrics.csv"
    audit = json.loads(audit_path.read_text())
    frame = pd.read_csv(csv_path)
    lead_frame = pd.read_csv(lead_csv_path)

    assert summary["status"] == "complete_for_current_compatible_cohort"
    assert summary["compatibility_audit_sha256"] == _sha256(audit_path)
    assert summary["benchmark_code_sha256"] == _sha256(code_path)
    assert summary["csv_sha256"] == _sha256(csv_path)
    assert summary["per_lead_case_metrics_csv_sha256"] == _sha256(lead_csv_path)
    assert summary["models_expected"] == audit["counts"]["compatible"]
    assert summary["models_completed"] == len(frame)
    assert set(frame["model_id"]) == {
        row["model_id"] for row in audit["models"] if row["compatible"]
    }
    expected_digests = {
        row["model_id"]: row["checkpoint_sha256"]
        for row in audit["models"]
        if row["compatible"]
    }
    assert frame.set_index("model_id")["checkpoint_sha256"].to_dict() == expected_digests
    assert frame["finite"].all()
    assert frame["output_shape"].eq("1x12x5000").all()
    assert summary["all_finite"] is True
    assert summary["cache_retained_bytes"] == 0
    assert len(lead_frame) == summary["models_completed"] * 9
    assert lead_frame.groupby("model_id").size().eq(9).all()
    assert set(lead_frame["lead"]) == {
        "III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6"
    }
    assert lead_frame[["mse", "mae", "pearson", "variance_ratio"]].notna().all().all()


def test_current_artifact_fast_path_detects_no_work():
    class Args:
        output_dir = RESULT_ROOT
        compatibility_audit = ROOT / "results/checkpoint_store/compatibility_audit.json"
        input = ROOT / "data/ptb_xl/tensors/test/100.pt"

    assert artifact_is_current(Args())
