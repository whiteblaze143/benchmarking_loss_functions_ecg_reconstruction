import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_temporal_target_detector_eda import (
    DEFAULT_METADATA,
    build,
    build_subgroup_coverage,
    load_test_metadata,
    validate_cache,
)


def _cache(tmp_path):
    frame = pd.DataFrame(
        [
            (1, "V3", "R_Amp", 0, 100, 0.8),
            (1, "V3", "R_Amp", 1, 500, 0.9),
            (2, "V3", "R_Amp", 0, 120, 1.0),
            (1, "V6", "QT_Interval_ms", 0, 80, 410.0),
        ],
        columns=[
            "record_id", "lead", "clinical_feature", "beat_index",
            "sample_index", "value",
        ],
    )
    parquet = tmp_path / "_target_ptb_xl_features.parquet"
    frame.to_parquet(parquet, index=False)
    metadata = {
        "schema_version": 2,
        "records": 2_198,
        "rows": len(frame),
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
    }
    (tmp_path / "_target_ptb_xl_features.json").write_text(json.dumps(metadata))
    return frame, parquet


def test_target_detector_eda_validates_and_summarizes_real_denominator(tmp_path):
    _, _ = _cache(tmp_path)
    frame, metadata, _, _ = validate_cache(tmp_path)
    result = build(frame, metadata["records"])
    r_amp = result[
        result["clinical_feature"].eq("R_Amp") & result["lead"].eq("V3")
    ].iloc[0]
    assert r_amp["records_total"] == 2_198
    assert r_amp["records_detected"] == 2
    assert r_amp["record_detection_coverage"] == pytest.approx(2 / 2_198)
    assert r_amp["events"] == 3
    assert r_amp["events_per_detected_record_median"] == 1.5
    assert r_amp["inter_event_interval_ms_median"] == 800.0
    assert np.isfinite(r_amp["value_median"])


def test_target_detector_eda_rejects_parquet_mutation(tmp_path):
    _, parquet = _cache(tmp_path)
    parquet.write_bytes(parquet.read_bytes() + b"mutation")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        validate_cache(tmp_path)


def test_target_detector_eda_rejects_duplicate_feature_event(tmp_path):
    frame, parquet = _cache(tmp_path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_parquet(parquet, index=False)
    metadata_path = tmp_path / "_target_ptb_xl_features.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["rows"] = len(frame)
    metadata["parquet_sha256"] = hashlib.sha256(parquet.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_cache(tmp_path)


def test_target_detector_subgroup_denominators_close_to_test_cohort(tmp_path):
    frame, _ = _cache(tmp_path)
    metadata = pd.DataFrame({
        "record_id": [str(index) for index in range(1, 2_199)],
        "sex_group": [f"sex_code_{index % 2}" for index in range(1, 2_199)],
        "age_group": ["age_60_79"] * 2_198,
    })
    result = build_subgroup_coverage(frame, metadata)
    denominators = result.groupby(
        ["lead", "clinical_feature", "subgroup_variable"]
    )["records"].sum()
    assert denominators.eq(2_198).all()
    v3_r_sex = result[
        result["lead"].eq("V3")
        & result["clinical_feature"].eq("R_Amp")
        & result["subgroup_variable"].eq("sex_group")
    ]
    assert v3_r_sex["records_detected"].sum() == 2


def test_actual_ptbxl_age_bins_isolate_invalid_values():
    metadata = load_test_metadata(DEFAULT_METADATA)
    counts = metadata["age_group"].value_counts().to_dict()
    assert len(metadata) == 2_198
    assert counts == {
        "age_60_79": 933,
        "age_40_59": 641,
        "age_80_100": 306,
        "age_<40": 284,
        "age_invalid_>100": 34,
    }
