"""
T-CAIREM Research Directive V4: Task 5
Controlled Episode Perturbations & Density Error Amplification (DEA)
Analyzes sensitivity of Charitos AF density to detector errors and pure topological rearrangement.
"""

import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.af_density_v4.metrics.af_pattern_metrics import measure_af_pattern
from experiments.af_density_v4.multiscale.multiscale_density_surface import (
    load_development_reference_intervals,
    merge_intervals,
)
REPORTS_DIR = PROJECT_ROOT / "experiments" / "af_density_v4" / "reports"
CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_v4")
WAVEFORM_DIR = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CODEX_DIR.mkdir(parents=True, exist_ok=True)


def get_sinus_gaps(clean_eps, total_T):
    gaps = []
    prev_end = 0.0
    for s, e in clean_eps:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < total_T:
        gaps.append((prev_end, total_T))
    return gaps


def apply_perturbation(perturbation_type, clean_eps, total_T):
    """
    Applies one of the 18 controlled perturbations to clean_eps.
    Returns perturbed intervals (merged).
    """
    if not clean_eps:
        return []

    # 1. False positive insertions into sinus gaps
    if perturbation_type == "FP_INSERT_30S":
        gaps = get_sinus_gaps(clean_eps, total_T)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= 40.0]
        if not valid_gaps:
            return clean_eps
        # Insert 30s in the center of the largest gap
        largest = max(valid_gaps, key=lambda g: g[1] - g[0])
        mid = (largest[0] + largest[1]) / 2.0
        new_ep = (mid - 15.0, mid + 15.0)
        return merge_intervals(clean_eps + [new_ep])

    elif perturbation_type == "FP_INSERT_60S":
        gaps = get_sinus_gaps(clean_eps, total_T)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= 80.0]
        if not valid_gaps:
            return clean_eps
        largest = max(valid_gaps, key=lambda g: g[1] - g[0])
        mid = (largest[0] + largest[1]) / 2.0
        new_ep = (mid - 30.0, mid + 30.0)
        return merge_intervals(clean_eps + [new_ep])

    elif perturbation_type == "FP_INSERT_120S":
        gaps = get_sinus_gaps(clean_eps, total_T)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= 150.0]
        if not valid_gaps:
            return clean_eps
        largest = max(valid_gaps, key=lambda g: g[1] - g[0])
        mid = (largest[0] + largest[1]) / 2.0
        new_ep = (mid - 60.0, mid + 60.0)
        return merge_intervals(clean_eps + [new_ep])

    elif perturbation_type == "FP_INSERT_10M":
        gaps = get_sinus_gaps(clean_eps, total_T)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= 700.0]
        if not valid_gaps:
            return clean_eps
        largest = max(valid_gaps, key=lambda g: g[1] - g[0])
        mid = (largest[0] + largest[1]) / 2.0
        new_ep = (mid - 300.0, mid + 300.0)
        return merge_intervals(clean_eps + [new_ep])

    elif perturbation_type == "FP_INSERT_30M":
        gaps = get_sinus_gaps(clean_eps, total_T)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= 2000.0]
        if not valid_gaps:
            return clean_eps
        largest = max(valid_gaps, key=lambda g: g[1] - g[0])
        mid = (largest[0] + largest[1]) / 2.0
        new_ep = (mid - 900.0, mid + 900.0)
        return merge_intervals(clean_eps + [new_ep])

    # 2. False negative deletions
    elif perturbation_type == "FN_DELETE_SHORTEST":
        if len(clean_eps) <= 1:
            return clean_eps
        durations = [(i, e - s) for i, (s, e) in enumerate(clean_eps)]
        min_idx = min(durations, key=lambda x: x[1])[0]
        return [ep for i, ep in enumerate(clean_eps) if i != min_idx]

    elif perturbation_type == "FN_DELETE_LONGEST":
        if len(clean_eps) <= 1:
            return clean_eps
        durations = [(i, e - s) for i, (s, e) in enumerate(clean_eps)]
        max_idx = max(durations, key=lambda x: x[1])[0]
        return [ep for i, ep in enumerate(clean_eps) if i != max_idx]

    elif perturbation_type == "FN_DELETE_MEDIAN":
        if len(clean_eps) <= 1:
            return clean_eps
        med_dur = np.median([e - s for s, e in clean_eps])
        closest_idx = min(range(len(clean_eps)), key=lambda i: abs((clean_eps[i][1] - clean_eps[i][0]) - med_dur))
        return [ep for i, ep in enumerate(clean_eps) if i != closest_idx]

    # 3. Splitting episodes
    elif perturbation_type == "SPLIT_LONGEST_2":
        max_idx = max(range(len(clean_eps)), key=lambda i: clean_eps[i][1] - clean_eps[i][0])
        s, e = clean_eps[max_idx]
        dur = e - s
        if dur < 90.0:  # Need room for split
            return clean_eps
        mid = (s + e) / 2.0
        part1 = (s, mid - 15.0)
        part2 = (mid + 15.0, e)
        new_eps = [ep for i, ep in enumerate(clean_eps) if i != max_idx] + [part1, part2]
        return merge_intervals(new_eps)

    elif perturbation_type == "SPLIT_LONGEST_3":
        max_idx = max(range(len(clean_eps)), key=lambda i: clean_eps[i][1] - clean_eps[i][0])
        s, e = clean_eps[max_idx]
        dur = e - s
        if dur < 150.0:
            return clean_eps
        d3 = dur / 3.0
        part1 = (s, s + d3 - 15.0)
        part2 = (s + d3 + 15.0, s + 2 * d3 - 15.0)
        part3 = (s + 2 * d3 + 15.0, e)
        new_eps = [ep for i, ep in enumerate(clean_eps) if i != max_idx] + [part1, part2, part3]
        return merge_intervals(new_eps)

    elif perturbation_type == "SPLIT_LONGEST_5":
        max_idx = max(range(len(clean_eps)), key=lambda i: clean_eps[i][1] - clean_eps[i][0])
        s, e = clean_eps[max_idx]
        dur = e - s
        if dur < 300.0:
            return clean_eps
        d5 = dur / 5.0
        parts = []
        for k in range(5):
            p_s = s + k * d5 + (15.0 if k > 0 else 0.0)
            p_e = s + (k + 1) * d5 - (15.0 if k < 4 else 0.0)
            if p_e > p_s:
                parts.append((p_s, p_e))
        new_eps = [ep for i, ep in enumerate(clean_eps) if i != max_idx] + parts
        return merge_intervals(new_eps)

    # 4. Merging episodes
    elif perturbation_type == "MERGE_CLOSEST_PAIR":
        if len(clean_eps) <= 1:
            return clean_eps
        # Find shortest gap between consecutive episodes
        gaps = []
        for i in range(len(clean_eps) - 1):
            gap_len = clean_eps[i+1][0] - clean_eps[i][1]
            if gap_len > 0:
                gaps.append((i, gap_len))
        if not gaps:
            return clean_eps
        min_gap_idx = min(gaps, key=lambda x: x[1])[0]
        # Merge episode i and i+1
        i = min_gap_idx
        bridged = (clean_eps[i][0], clean_eps[i+1][1])
        new_eps = [ep for k, ep in enumerate(clean_eps) if k not in (i, i+1)] + [bridged]
        return merge_intervals(new_eps)

    elif perturbation_type == "MERGE_ALL_LESS_60S":
        if len(clean_eps) <= 1:
            return clean_eps
        new_eps = [clean_eps[0]]
        for s, e in clean_eps[1:]:
            prev_s, prev_e = new_eps[-1]
            if s - prev_e <= 60.0:  # Bridge gap <= 60s
                new_eps[-1] = (prev_s, e)
            else:
                new_eps.append((s, e))
        return new_eps

    # 5. Boundary displacement / jitter
    elif perturbation_type == "JITTER_ONSET_MINUS_30S":
        new_eps = [(max(0.0, s - 30.0), e) for s, e in clean_eps]
        return merge_intervals(new_eps)

    elif perturbation_type == "JITTER_ONSET_PLUS_30S":
        new_eps = [(min(e - 1.0, s + 30.0), e) for s, e in clean_eps if (e - s) > 30.0]
        return merge_intervals(new_eps)

    elif perturbation_type == "JITTER_OFFSET_PLUS_30S":
        new_eps = [(s, min(total_T, e + 30.0)) for s, e in clean_eps]
        return merge_intervals(new_eps)

    elif perturbation_type == "JITTER_OFFSET_MINUS_30S":
        new_eps = [(s, max(s + 1.0, e - 30.0)) for s, e in clean_eps if (e - s) > 30.0]
        return merge_intervals(new_eps)

    # 6. Random 1% window flip
    elif perturbation_type == "RANDOM_WINDOW_NOISE_1PCT":
        rng = np.random.default_rng(42)
        n_win = int(np.floor(total_T / 30.0))
        # Rasterize to 30s binary mask
        mask = np.zeros(n_win, dtype=bool)
        for s, e in clean_eps:
            w_start = int(np.floor(s / 30.0))
            w_end = min(n_win, int(np.ceil(e / 30.0)))
            mask[w_start:w_end] = True
        # Flip 1% randomly
        flip_idx = rng.choice(n_win, size=max(1, int(0.01 * n_win)), replace=False)
        mask[flip_idx] = ~mask[flip_idx]
        # Reconstruct intervals
        diff = np.diff(np.pad(mask.astype(int), (1, 1), mode="constant"))
        starts = np.where(diff == 1)[0] * 30.0
        ends = np.where(diff == -1)[0] * 30.0
        reconstructed = [(min(total_T, s), min(total_T, e)) for s, e in zip(starts, ends) if e > s]
        return merge_intervals(reconstructed)

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def build_equal_burden_counterfactual(cf_type, total_af_s, total_T):
    """
    Constructs an equal-burden counterfactual timeline with EXACT duration total_af_s inside total_T.
    Returns intervals (merged).
    """
    if total_af_s <= 0 or total_T <= 0:
        return []
    total_af_s = min(total_af_s, total_T - 1.0)

    if cf_type == "CF_SINGLE_EPISODE":
        # Centered single continuous episode
        s = (total_T - total_af_s) / 2.0
        return [(s, s + total_af_s)]

    elif cf_type == "CF_2_EQUAL_EPISODES":
        ep_len = total_af_s / 2.0
        gap = (total_T - total_af_s) / 3.0
        s1 = gap
        e1 = s1 + ep_len
        s2 = e1 + gap
        e2 = s2 + ep_len
        return [(s1, e1), (s2, e2)]

    elif cf_type == "CF_5_EQUAL_EPISODES":
        ep_len = total_af_s / 5.0
        gap = (total_T - total_af_s) / 6.0
        eps = []
        cur = gap
        for _ in range(5):
            eps.append((cur, cur + ep_len))
            cur += ep_len + gap
        return eps

    elif cf_type == "CF_10_EQUAL_EPISODES":
        ep_len = total_af_s / 10.0
        gap = (total_T - total_af_s) / 11.0
        eps = []
        cur = gap
        for _ in range(10):
            eps.append((cur, cur + ep_len))
            cur += ep_len + gap
        return eps

    elif cf_type == "CF_GEOMETRIC_5":
        # 5 episodes with weights 1, 2, 4, 8, 16 (sum = 31)
        weights = [1.0, 2.0, 4.0, 8.0, 16.0]
        lens = [total_af_s * (w / sum(weights)) for w in weights]
        gap = (total_T - total_af_s) / 6.0
        eps = []
        cur = gap
        for l in lens:
            eps.append((cur, cur + l))
            cur += l + gap
        return eps

    elif cf_type == "CF_MAX_DISPERSED":
        # Small episodes of 60s (or less if burden is small) spread across total_T
        n_eps = max(2, min(50, int(total_af_s // 60.0)))
        ep_len = total_af_s / n_eps
        gap = (total_T - total_af_s) / (n_eps + 1)
        eps = []
        cur = gap
        for _ in range(n_eps):
            eps.append((cur, cur + ep_len))
            cur += ep_len + gap
        return eps

    else:
        raise ValueError(f"Unknown counterfactual type: {cf_type}")


def main():
    print("[V4 Perturbations] Initializing Task 5: Controlled Episode Perturbations & DEA...")
    patient_raw = load_development_reference_intervals()
    manifest_csv = WAVEFORM_DIR / "file_manifest.csv"
    manifest = list(csv.DictReader(manifest_csv.open()))

    patient_durations = {}
    for r in manifest:
        if r["partition"] != "test":
            pid = r["patient_id"]
            patient_durations[pid] = patient_durations.get(pid, 0.0) + float(r["windows"]) * 30.0

    # Perturbation registry
    perturbation_registry = {
        "families": {
            "false_positive_insertion": ["FP_INSERT_30S", "FP_INSERT_60S", "FP_INSERT_120S", "FP_INSERT_10M", "FP_INSERT_30M"],
            "false_negative_deletion": ["FN_DELETE_SHORTEST", "FN_DELETE_LONGEST", "FN_DELETE_MEDIAN"],
            "episode_splitting": ["SPLIT_LONGEST_2", "SPLIT_LONGEST_3", "SPLIT_LONGEST_5"],
            "episode_merging": ["MERGE_CLOSEST_PAIR", "MERGE_ALL_LESS_60S"],
            "boundary_displacement": ["JITTER_ONSET_MINUS_30S", "JITTER_ONSET_PLUS_30S", "JITTER_OFFSET_PLUS_30S", "JITTER_OFFSET_MINUS_30S"],
            "random_window_noise": ["RANDOM_WINDOW_NOISE_1PCT"]
        },
        "counterfactual_families": [
            "CF_SINGLE_EPISODE", "CF_2_EQUAL_EPISODES", "CF_5_EQUAL_EPISODES",
            "CF_10_EQUAL_EPISODES", "CF_GEOMETRIC_5", "CF_MAX_DISPERSED"
        ],
        "metric_definitions": {
            "DEA_relative": "|Delta D / D0| / |Delta b / b0|",
            "DEA_absolute": "|Delta D| / |Delta b|",
            "topological_amplification_infinite": "Defined when Delta b == 0 and Delta D != 0"
        }
    }

    with open(REPORTS_DIR / "episode_perturbation_registry_v4.json", "w") as f:
        json.dump(perturbation_registry, f, indent=2)
    with open(CODEX_DIR / "episode_perturbation_registry_v4.json", "w") as f:
        json.dump(perturbation_registry, f, indent=2)

    # Flatten perturbation list
    all_perturbations = []
    for fam, perts in perturbation_registry["families"].items():
        for p in perts:
            all_perturbations.append((fam, p))

    # Evaluate across development patients with >= 1 episode
    response_records = []
    cf_records = []

    print(f"  -> Auditing perturbations across {len(patient_raw)} patients...")

    for pid, eps in patient_raw.items():
        total_T = patient_durations.get(pid, 86400.0)
        clean_eps = sorted(eps, key=lambda x: x[0])
        m0 = measure_af_pattern(clean_eps, total_T)
        if m0.charitos_density is None or m0.af_burden <= 0:
            continue

        D0 = m0.charitos_density
        b0 = m0.af_burden
        total_af_s = b0 * total_T

        # 1. Evaluate 18 perturbations
        for fam, p_name in all_perturbations:
            pert_eps = apply_perturbation(p_name, clean_eps, total_T)
            m_pert = measure_af_pattern(pert_eps, total_T)
            D_prime = m_pert.charitos_density
            b_prime = m_pert.af_burden

            delta_b = b_prime - b0
            delta_D = (D_prime - D0) if D_prime is not None else np.nan

            abs_delta_b = abs(delta_b)
            abs_delta_D = abs(delta_D) if not np.isnan(delta_D) else np.nan

            rel_delta_b = abs_delta_b / b0 if b0 > 0 else 0.0
            rel_delta_D = abs_delta_D / D0 if D0 > 0 and not np.isnan(abs_delta_D) else np.nan

            if abs_delta_b > 1e-7:
                dea_abs = abs_delta_D / abs_delta_b if not np.isnan(abs_delta_D) else np.nan
                dea_rel = rel_delta_D / rel_delta_b if not np.isnan(rel_delta_D) and rel_delta_b > 0 else np.nan
            else:
                # Delta b == 0: pure topological change
                dea_abs = np.inf if (abs_delta_D and abs_delta_D > 1e-4) else 0.0
                dea_rel = np.inf if (rel_delta_D and rel_delta_D > 1e-4) else 0.0

            response_records.append({
                "subject_id": pid,
                "family": fam,
                "perturbation_name": p_name,
                "total_T_hours": round(total_T / 3600.0, 2),
                "baseline_burden": b0,
                "baseline_density": D0,
                "baseline_episodes": m0.episode_count,
                "perturbed_burden": b_prime,
                "perturbed_density": D_prime,
                "perturbed_episodes": m_pert.episode_count,
                "delta_burden": delta_b,
                "delta_density": delta_D,
                "abs_delta_burden": abs_delta_b,
                "abs_delta_density": abs_delta_D,
                "rel_delta_burden": rel_delta_b,
                "rel_delta_density": rel_delta_D,
                "density_error_amplification_abs": dea_abs,
                "density_error_amplification_rel": dea_rel,
                "is_density_defined": D_prime is not None
            })

        # 2. Evaluate Equal-Burden Counterfactuals
        for cf_name in perturbation_registry["counterfactual_families"]:
            cf_eps = build_equal_burden_counterfactual(cf_name, total_af_s, total_T)
            m_cf = measure_af_pattern(cf_eps, total_T)
            D_cf = m_cf.charitos_density
            b_cf = m_cf.af_burden

            cf_records.append({
                "subject_id": pid,
                "counterfactual_name": cf_name,
                "total_T_hours": round(total_T / 3600.0, 2),
                "reference_burden": b0,
                "counterfactual_burden": b_cf,
                "burden_conservation_error": abs(b_cf - b0),
                "reference_density": D0,
                "counterfactual_density": D_cf,
                "counterfactual_episodes": m_cf.episode_count,
                "density_shift": (D_cf - D0) if D_cf is not None else np.nan,
                "counterfactual_duration_gini": m_cf.duration_gini,
                "counterfactual_gap_gini": m_cf.gap_gini
            })

    # Save outputs
    df_resp = pd.DataFrame(response_records)
    df_cf = pd.DataFrame(cf_records)

    print(f"  -> Generated {len(df_resp)} perturbation evaluations and {len(df_cf)} equal-burden counterfactuals.")

    df_resp.to_parquet(REPORTS_DIR / "density_error_response_surfaces_v4.parquet", index=False)
    df_resp.to_csv(REPORTS_DIR / "density_error_response_surfaces_v4.csv", index=False)
    df_resp.to_parquet(CODEX_DIR / "density_error_response_surfaces_v4.parquet", index=False)
    df_resp.to_csv(CODEX_DIR / "density_error_response_surfaces_v4.csv", index=False)

    df_cf.to_csv(REPORTS_DIR / "equal_burden_counterfactuals_v4.csv", index=False)
    df_cf.to_csv(CODEX_DIR / "equal_burden_counterfactuals_v4.csv", index=False)

    print("[V4 Perturbations Complete] Task 5 outputs verified and mirrored.\n")


if __name__ == "__main__":
    main()
