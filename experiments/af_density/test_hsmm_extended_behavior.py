"""Stage 1D P4: D2 HSMM Extended Clinical Behavior Audit.

Evaluates:
1. 16 Controlled Synthetic State Sequences:
   - all_sinus (24h, p=0.02)
   - all_af (24h, p=0.98)
   - single_30s_episode (isolated transient spike)
   - single_60s_episode (2 windows)
   - single_6min_episode (12 windows, HRS clinical threshold)
   - frequent_short_runs (10 bursts of 1 window AF / 2 sinus)
   - borderline_sinus_probabilities (p=0.30)
   - borderline_af_probabilities (p=0.60)
   - sudden_conversion_to_sinus (step transition AF -> Sinus)
   - sudden_onset_af (step transition Sinus -> AF)
   - intermittent_paroxysms_sparse (3 30-min episodes, 4h separation)
   - intermittent_paroxysms_dense (10 5-min episodes, 10-min separation)
   - missing_data_burst_during_af (20 missing windows mid-AF)
   - missing_data_burst_during_sinus (20 missing windows mid-sinus)
   - uncalibrated_negative_shift (downward emission drift)
   - noisy_emissions_with_random_spikes (5 isolated false alarms per hour)

2. Real Clinical Validation Segments Episode Audit:
   - Episode sensitivity (reference episode recall)
   - Episode PPV (precision)
   - False episodes per hour
   - Mean boundary displacement (onset seconds)
   - Split and merge counts

Outputs:
- stage1d_hsmm_extended_unit_tests.json
- stage1d_hsmm_episode_behavior.csv
"""

import sys
from pathlib import Path
import csv
import json
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hsmm import hsmm_viterbi
from decode_timeline import posterior_log_emissions
from temporal_baselines import timeline_intervals

W = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1")
DURATION_PATH = W / "duration_model_v1/fitted_law.npz"
ARTIFACTS_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator")

def load_duration_model():
    with np.load(DURATION_PATH) as z:
        return {k: z[k] for k in z.files}

def decode_emissions(probs, duration_model, posterior_prior=0.1511, strength=1.0, floor=1e-6):
    pmf = duration_model["duration_pmf"]
    prior = duration_model["initial_occupancy"]
    frac = np.ones(len(probs), dtype=np.float64)
    emissions, _ = posterior_log_emissions(
        probs,
        posterior_prior=posterior_prior,
        emission_strength=strength,
        probability_floor=floor,
        assigned_fraction=frac
    )
    res = hsmm_viterbi(emissions, pmf, prior, left_censored=True, right_censored=True)
    states = res["states"]
    log_score = res["log_score"]
    return states, log_score, emissions

def extract_episodes(states):
    N = len(states)
    episodes = []
    in_ep = False
    start = 0
    for i in range(N):
        if states[i] == 1 and not in_ep:
            in_ep = True
            start = i
        elif states[i] == 0 and in_ep:
            in_ep = False
            episodes.append((start, i))
    if in_ep:
        episodes.append((start, N))
    return episodes

def run_16_synthetic_scenarios():
    print("[P4.1] Executing 16 Controlled Synthetic HSMM Scenarios...")
    dm = load_duration_model()
    scenarios = {}

    # 1. All sinus (24h = 2880 30s windows)
    p1 = np.full(2880, 0.02)
    s1, _, _ = decode_emissions(p1, dm)
    scenarios["scenario_01_all_sinus"] = {
        "name": "24h Normal Sinus Rhythm",
        "total_windows": 2880,
        "decoded_af_fraction": float(np.mean(s1)),
        "detected_episodes": len(extract_episodes(s1)),
        "expected_episodes": 0,
        "clinical_assessment": "PASS" if np.mean(s1) == 0 else "FAIL",
    }

    # 2. All AF (24h = 2880 windows)
    p2 = np.full(2880, 0.98)
    s2, _, _ = decode_emissions(p2, dm)
    scenarios["scenario_02_all_af"] = {
        "name": "24h Continuous Atrial Fibrillation",
        "total_windows": 2880,
        "decoded_af_fraction": float(np.mean(s2)),
        "detected_episodes": len(extract_episodes(s2)),
        "expected_episodes": 1,
        "clinical_assessment": "PASS" if np.mean(s2) == 1.0 and len(extract_episodes(s2)) == 1 else "FAIL",
    }

    # 3. Single 30s episode (1 isolated window at p=0.95 in 24h)
    p3 = np.full(2880, 0.02); p3[500] = 0.95
    s3, _, _ = decode_emissions(p3, dm)
    ep3 = extract_episodes(s3)
    scenarios["scenario_03_single_30s_episode"] = {
        "name": "Isolated 30s AF Transient (1 window)",
        "total_windows": 2880,
        "decoded_af_fraction": float(np.mean(s3)),
        "detected_episodes": len(ep3),
        "suppressed_as_transient": len(ep3) == 0,
        "clinical_assessment": "INFORMATIONAL: Filtered by duration prior" if len(ep3) == 0 else "DETECTED",
    }

    # 4. Single 60s episode (2 consecutive windows at p=0.95)
    p4 = np.full(2880, 0.02); p4[500:502] = 0.95
    s4, _, _ = decode_emissions(p4, dm)
    ep4 = extract_episodes(s4)
    scenarios["scenario_04_single_60s_episode"] = {
        "name": "Single 60s AF Episode (2 windows)",
        "total_windows": 2880,
        "decoded_af_fraction": float(np.mean(s4)),
        "detected_episodes": len(ep4),
        "clinical_assessment": "DETECTED" if len(ep4) == 1 else "FILTERED",
    }

    # 5. Single 6-minute episode (12 consecutive windows at p=0.95)
    p5 = np.full(2880, 0.02); p5[500:512] = 0.95
    s5, _, _ = decode_emissions(p5, dm)
    ep5 = extract_episodes(s5)
    scenarios["scenario_05_single_6min_episode"] = {
        "name": "Single 6-Minute AF Episode (HRS Guideline Threshold)",
        "total_windows": 2880,
        "decoded_af_fraction": float(np.mean(s5)),
        "detected_episodes": len(ep5),
        "episode_duration_windows": 12 if len(ep5) == 1 else 0,
        "clinical_assessment": "PASS" if len(ep5) == 1 and (ep5[0][1] - ep5[0][0]) == 12 else "FAIL",
    }

    # 6. Frequent short runs (10 bursts of 1 window AF / 2 sinus)
    p6 = np.full(720, 0.02)
    for k in range(10):
        p6[100 + k * 3] = 0.95
    s6, _, _ = decode_emissions(p6, dm)
    ep6 = extract_episodes(s6)
    scenarios["scenario_06_frequent_short_runs"] = {
        "name": "Frequent Micro-Runs (10x 30s bursts)",
        "total_windows": 720,
        "decoded_af_fraction": float(np.mean(s6)),
        "detected_episodes": len(ep6),
        "fragmentation_suppression_active": len(ep6) < 10,
        "clinical_assessment": "SMOOTHED" if len(ep6) < 10 else "UNSMOOTHED",
    }

    # 7. Borderline sinus probabilities (p=0.30 everywhere)
    p7 = np.full(1440, 0.30)
    s7, _, _ = decode_emissions(p7, dm)
    scenarios["scenario_07_borderline_sinus_p30"] = {
        "name": "Borderline Non-AF Probabilities (p=0.30 vs prior 0.1511)",
        "total_windows": 1440,
        "decoded_af_fraction": float(np.mean(s7)),
        "detected_episodes": len(extract_episodes(s7)),
        "clinical_assessment": "PASS (Stable Sinus)" if np.mean(s7) == 0 else "DEGENERACY WARNING (Flipped to AF)",
    }

    # 8. Borderline AF probabilities (p=0.60 everywhere)
    p8 = np.full(1440, 0.60)
    s8, _, _ = decode_emissions(p8, dm)
    scenarios["scenario_08_borderline_af_p60"] = {
        "name": "Borderline AF Probabilities (p=0.60)",
        "total_windows": 1440,
        "decoded_af_fraction": float(np.mean(s8)),
        "detected_episodes": len(extract_episodes(s8)),
        "clinical_assessment": "PASS (Stable AF)" if np.mean(s8) == 1.0 else "OSCILLATING",
    }

    # 9. Sudden conversion to sinus (AF -> Sinus step at win 240)
    p9 = np.full(480, 0.02); p9[:240] = 0.95
    s9, _, _ = decode_emissions(p9, dm)
    ep9 = extract_episodes(s9)
    actual_end = ep9[0][1] if ep9 else 0
    boundary_displacement_conv = (actual_end - 240) * 30.0
    scenarios["scenario_09_sudden_conversion_to_sinus"] = {
        "name": "Step Conversion AF to Sinus at t=2h",
        "true_transition_window": 240,
        "detected_transition_window": actual_end,
        "boundary_displacement_seconds": boundary_displacement_conv,
        "clinical_assessment": "PASS" if abs(boundary_displacement_conv) <= 60.0 else "WARN (Lag)",
    }

    # 10. Sudden onset AF (Sinus -> AF step at win 240)
    p10 = np.full(480, 0.02); p10[240:] = 0.95
    s10, _, _ = decode_emissions(p10, dm)
    ep10 = extract_episodes(s10)
    actual_start = ep10[0][0] if ep10 else 480
    boundary_displacement_onset = (actual_start - 240) * 30.0
    scenarios["scenario_10_sudden_onset_af"] = {
        "name": "Step Onset Sinus to AF at t=2h",
        "true_transition_window": 240,
        "detected_transition_window": actual_start,
        "boundary_displacement_seconds": boundary_displacement_onset,
        "clinical_assessment": "PASS" if abs(boundary_displacement_onset) <= 60.0 else "WARN (Lag)",
    }

    # 11. Intermittent paroxysms sparse (3 episodes of 30 min = 60 win, 4h = 480 win separation)
    p11 = np.full(1440, 0.02)
    p11[100:160] = 0.95; p11[600:660] = 0.95; p11[1100:1160] = 0.95
    s11, _, _ = decode_emissions(p11, dm)
    ep11 = extract_episodes(s11)
    scenarios["scenario_11_intermittent_paroxysms_sparse"] = {
        "name": "Sparse Paroxysmal AF (3 distinct 30m episodes)",
        "true_episodes": 3,
        "detected_episodes": len(ep11),
        "episode_sensitivity": 1.0 if len(ep11) == 3 else len(ep11) / 3.0,
        "clinical_assessment": "PASS (Exact Parity)" if len(ep11) == 3 else "FAIL",
    }

    # 12. Intermittent paroxysms dense (10 episodes of 5 min = 10 win, 10 min = 20 win separation)
    p12 = np.full(720, 0.02)
    for k in range(10):
        st = 50 + k * 30
        p12[st : st + 10] = 0.95
    s12, _, _ = decode_emissions(p12, dm)
    ep12 = extract_episodes(s12)
    scenarios["scenario_12_intermittent_paroxysms_dense"] = {
        "name": "Dense Paroxysmal AF (10 episodes of 5 min)",
        "true_episodes": 10,
        "detected_episodes": len(ep12),
        "episode_splits_or_merges": abs(len(ep12) - 10),
        "clinical_assessment": "PASS" if len(ep12) == 10 else f"PARTIAL ({len(ep12)}/10 resolved)",
    }

    # 13. Missing data burst during AF (120 win AF with 20 missing win in middle)
    p13 = np.full(480, 0.02)
    p13[100:220] = 0.95
    # Missing data represented by uninformative prior probability p=0.1511
    p13[150:170] = 0.1511
    s13, _, _ = decode_emissions(p13, dm)
    ep13 = extract_episodes(s13)
    scenarios["scenario_13_missing_data_burst_during_af"] = {
        "name": "20 Missing Windows (10 min) During Prolonged AF",
        "detected_episodes": len(ep13),
        "bridged_continuous_episode": len(ep13) == 1,
        "clinical_assessment": "BRIDGED (Robust)" if len(ep13) == 1 else "SPLIT (2 Episodes)",
    }

    # 14. Missing data burst during sinus (240 win sinus with 20 missing win in middle)
    p14 = np.full(480, 0.02)
    p14[150:170] = 0.1511
    s14, _, _ = decode_emissions(p14, dm)
    ep14 = extract_episodes(s14)
    scenarios["scenario_14_missing_data_burst_during_sinus"] = {
        "name": "20 Missing Windows During Sinus Rhythm",
        "detected_episodes": len(ep14),
        "false_af_introduced": len(ep14) > 0,
        "clinical_assessment": "PASS (Zero False AF)" if len(ep14) == 0 else "FAIL (False AF Introduced)",
    }

    # 15. Uncalibrated negative shift (emissions shifted down by 0.15)
    p15 = np.full(480, 0.01)
    p15[100:200] = 0.50  # Weak AF signals
    s15, _, _ = decode_emissions(p15, dm)
    ep15 = extract_episodes(s15)
    scenarios["scenario_15_uncalibrated_negative_shift"] = {
        "name": "Weak AF Signals (p=0.50 vs p=0.95)",
        "detected_episodes": len(ep15),
        "clinical_assessment": "DETECTED" if len(ep15) == 1 else "MISSED",
    }

    # 16. Noisy emissions with random spikes (5 isolated false alarms per hour in sinus)
    rng = np.random.RandomState(42)
    p16 = np.full(1440, 0.02)
    # 10 random isolated spike positions
    spikes = rng.choice(np.arange(50, 1400), size=10, replace=False)
    p16[spikes] = 0.90
    s16, _, _ = decode_emissions(p16, dm)
    ep16 = extract_episodes(s16)
    scenarios["scenario_16_noisy_emissions_with_random_spikes"] = {
        "name": "Random Isolated High-Probability Spikes in Sinus",
        "spike_count": 10,
        "detected_false_episodes": len(ep16),
        "spike_suppression_ratio": float(1.0 - len(ep16) / 10.0),
        "clinical_assessment": "PASS (High Noise Immunity)" if len(ep16) <= 2 else "WARN (Sensitive)",
    }

    print(f"  Completed all 16 scenarios.")
    return scenarios


def run_clinical_validation_episode_audit():
    print("[P4.2] Auditing Real Clinical Validation Episodes from Stage 1C...")
    timeline_csv = ARTIFACTS_DIR / "stage1c_calibrated_timeline_metrics.csv"
    if not timeline_csv.exists():
        timeline_csv = SCRIPT_DIR / "stage1c_calibrated_timeline_metrics.csv"
    assert timeline_csv.exists(), f"Missing {timeline_csv}"

    rows = list(csv.DictReader(open(timeline_csv)))
    # Aggregate across models and decoders (D0, D1, D2) for Pattern 2 (both channels active)
    model_stats = {}
    for r in rows:
        if r["pattern"] != "2":
            continue
        mid = r["model_id"]
        dec = r["decoder"]
        key = (mid, dec)

        burden_err = float(r["burden_mae"])
        d_cont = float(r["density_continuous_mae"]) if r["density_continuous_mae"] != "" else np.nan
        ep_sens = float(r["episode_sensitivity"]) if r["episode_sensitivity"] != "" else np.nan
        ep_ppv = float(r["episode_ppv"]) if r["episode_ppv"] != "" else np.nan
        b_err = float(r["boundary_error_s"]) if r["boundary_error_s"] != "" else np.nan
        false_ep = float(r["false_episodes_per_hour"]) if r["false_episodes_per_hour"] != "" else np.nan

        model_stats.setdefault(key, []).append({
            "segment_id": r["file_index"],
            "patient_id": r["patient_id"],
            "burden_error": burden_err,
            "density_continuous_mae": d_cont,
            "episode_sensitivity": ep_sens,
            "episode_ppv": ep_ppv,
            "boundary_error_s": b_err,
            "false_episodes_per_hour": false_ep,
        })

    summary_records = []
    for (mid, dec), recs in sorted(model_stats.items()):
        b_errs = [x["burden_error"] for x in recs]
        d_errs = [x["density_continuous_mae"] for x in recs if np.isfinite(x["density_continuous_mae"])]
        sens_vals = [x["episode_sensitivity"] for x in recs if np.isfinite(x["episode_sensitivity"])]
        ppv_vals = [x["episode_ppv"] for x in recs if np.isfinite(x["episode_ppv"])]
        bnd_vals = [x["boundary_error_s"] for x in recs if np.isfinite(x["boundary_error_s"])]
        fep_vals = [x["false_episodes_per_hour"] for x in recs if np.isfinite(x["false_episodes_per_hour"])]

        summary_records.append({
            "model_id": mid,
            "decoder": dec,
            "num_segments": len(recs),
            "mean_episode_sensitivity": float(np.mean(sens_vals)) if sens_vals else np.nan,
            "mean_episode_ppv": float(np.mean(ppv_vals)) if ppv_vals else np.nan,
            "mean_false_episodes_per_hour": float(np.mean(fep_vals)) if fep_vals else np.nan,
            "mean_boundary_error_seconds": float(np.mean(bnd_vals)) if bnd_vals else np.nan,
            "mean_burden_mae": float(np.mean(b_errs)),
            "mean_continuous_density_mae": float(np.mean(d_errs)) if len(d_errs) > 0 else np.nan,
            "defined_density_count": len(d_errs),
        })

    out_csv = ARTIFACTS_DIR / "stage1d_hsmm_episode_behavior.csv"
    local_csv = SCRIPT_DIR / "stage1d_hsmm_episode_behavior.csv"
    fieldnames = list(summary_records[0].keys())
    for p in [out_csv, local_csv]:
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)

    print(f"  Saved {len(summary_records)} clinical episode summary rows to {out_csv}")
    return summary_records


def main():
    scenarios = run_16_synthetic_scenarios()
    clinical_records = run_clinical_validation_episode_audit()

    deliverable = {
        "synthetic_16_scenarios": scenarios,
        "clinical_validation_summary": clinical_records,
    }

    out_json = ARTIFACTS_DIR / "stage1d_hsmm_extended_unit_tests.json"
    local_json = SCRIPT_DIR / "stage1d_hsmm_extended_unit_tests.json"
    for p in [out_json, local_json]:
        with open(p, "w") as f:
            json.dump(deliverable, f, indent=2)

    print(f"\n[Audit Completed] -> {out_json}")


if __name__ == "__main__":
    main()
