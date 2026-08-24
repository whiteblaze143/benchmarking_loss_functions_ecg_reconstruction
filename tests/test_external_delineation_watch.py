import numpy as np
import pandas as pd

from scripts.evaluate_external_delineation_watch import (
    TARGET_SAMPLES,
    bounds_from_mask,
    monotonic_match,
    resample_exact,
    scale_indices,
    summarize,
)


def test_monotonic_match_is_one_to_one_and_maximizes_count():
    # Both reference events are near the first prediction. A nearest-neighbour
    # loop could count that prediction twice; the production matcher cannot.
    pairs = monotonic_match([100, 110], [105], tolerance_samples=10)
    assert len(pairs) == 1
    assert len({predicted for _, predicted in pairs}) == 1


def test_monotonic_match_prefers_lower_total_error_after_cardinality():
    pairs = monotonic_match([100, 200], [90, 101, 201], tolerance_samples=20)
    assert pairs == [(100, 101), (200, 201)]


def test_scale_indices_maps_native_endpoint_into_model_domain():
    result = scale_indices([0, 10_000, 19_999], source_samples=20_000)
    assert result.tolist() == [0, 2500, 4999]


def test_resample_exact_corrects_odd_length_records():
    signal = np.tile(np.linspace(-1, 1, 9_999), (12, 1))
    result = resample_exact(signal, up=1, down=2)
    assert result.shape == (12, TARGET_SAMPLES)
    assert np.isfinite(result).all()


def test_bounds_from_mask_preserves_regions_and_boundaries():
    mask = np.zeros(20_000)
    mask[100:200] = 1
    mask[300:500] = 2
    mask[700:1000] = 3
    bounds, masks = bounds_from_mask(mask)
    assert bounds["P_Onset"].tolist() == [25]
    assert bounds["P_Offset"].tolist() == [49]
    assert bounds["R_Onset"].tolist() == [75]
    assert bounds["R_Offset"].tolist() == [124]
    assert bounds["T_Onset"].tolist() == [175]
    assert bounds["T_Offset"].tolist() == [249]
    assert masks["P"].sum() == 25
    assert masks["QRS"].sum() == 50
    assert masks["T"].sum() == 75


def test_summary_reports_micro_f1_from_event_counts():
    events = pd.DataFrame(
        [
            {
                "dataset": "ludb",
                "split": "all",
                "lead_role": "missing",
                "boundary": "R_Onset",
                "record_id": "2",
                "detector_status": "ok",
                "reference_events": 3,
                "predicted_events": 3,
                "true_positive": 2,
                "false_positive": 1,
                "false_negative": 1,
                "f1": 2 / 3,
                "timing_mae_ms_header_derived": 4.0,
            },
            {
                "dataset": "ludb",
                "split": "all",
                "lead_role": "missing",
                "boundary": "R_Onset",
                "record_id": "3",
                "detector_status": "detector_error:ValueError",
                "reference_events": 2,
                "predicted_events": 0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 2,
                "f1": 0.0,
                "timing_mae_ms_header_derived": np.nan,
            },
        ]
    )
    overlaps = pd.DataFrame(
        [
            {
                "dataset": "ludb",
                "split": "all",
                "lead_role": "missing",
                "wave": "QRS",
                "record_id": "2",
                "detector_status": "ok",
                "dice": 0.8,
            }
        ]
    )
    result = summarize(events, overlaps)
    boundary = result.loc[result.metric_family == "boundary"].iloc[0]
    assert boundary.micro_f1 == 4 / 8
    assert boundary.detector_success_rate == 0.5
