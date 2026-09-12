#!/usr/bin/env python3
"""Generate the canonical REFERENCE_REPRESENTATION_SCHEMA.json for Milestone M5.

Documents every individual feature across all six baseline representations:
- R0_basic (23D): Standard physical VCG field statistics
- R0_rich (101D): Capacity-matched physical VCG control
- R1 (67D): Spatial domain occupancy histogram & categorical dynamics
- R2_H (68D): Hilbert functional trajectory statistics
- R3 (101D): Coupled physical + Hilbert representation
- R4 (122D): Full canonical reference representation Z_{12}^*
"""
from __future__ import annotations

import json
from pathlib import Path


def build_schema() -> dict:
    schema = {
        "title": "PTB-XL Canonical Deterministic ECG Reference Representation Schema (Z_{12}^*)",
        "milestone": "M5_REFERENCE_REPRESENTATION_LOCKDOWN",
        "version": "1.0.0",
        "date": "2026-09-12",
        "primary_equation": "Z_{12}^* = R4(X_{12}) = [R0_basic, R2_H, Coupling_vh, R_residual] in R^{122}",
        "matched_control_hypothesis": "R3 (101D) vs R0_rich (101D)",
        "baselines_summary": {
            "R0_basic": {"dimension": 23, "description": "Standard physical VCG dipole field statistics"},
            "R0_rich": {"dimension": 101, "description": "Capacity-matched physical-only VCG control"},
            "R1": {"dimension": 67, "description": "Spatial-domain categorical occupancy & dwell statistics"},
            "R2_H": {"dimension": 68, "description": "Hilbert functional trajectory statistics from h_H(t)"},
            "R3": {"dimension": 101, "description": "Coupled physical + Hilbert functional representation"},
            "R4": {"dimension": 122, "description": "Full canonical reference representation including non-dipolar residual"},
        },
        "features": []
    }

    # Helper to add feature
    feat_id = 0

    def add_feat(name, group, formula, input_source, units, description, sub_indices=None):
        nonlocal feat_id
        feat_id += 1
        entry = {
            "id": feat_id,
            "name": name,
            "baseline_memberships": group,
            "formula": formula,
            "input_source": input_source,
            "units": units,
            "description": description,
        }
        if sub_indices:
            entry["baseline_indices"] = sub_indices
        schema["features"].append(entry)

    # 1. Physical VCG Base Features (23 features: in R0_basic, R0_rich, R3, R4)
    # Mean (3)
    for ax in ["x", "y", "z"]:
        add_feat(f"v_mean_{ax}", ["R0_basic", "R0_rich", "R3", "R4"], f"mean(v_{ax}(t))", "v(t)", "mV", f"Temporal mean of physical dipole coordinate {ax}")
    # Covariance (6)
    for pair in ["xx", "xy", "xz", "yy", "yz", "zz"]:
        add_feat(f"v_cov_{pair}", ["R0_basic", "R0_rich", "R3", "R4"], f"cov(v_{pair[0]}, v_{pair[1]})", "v(t)", "mV^2", f"Covariance of physical coordinates {pair[0]} and {pair[1]}")
    # Range (3)
    for ax in ["x", "y", "z"]:
        add_feat(f"v_range_{ax}", ["R0_basic", "R0_rich", "R3", "R4"], f"ptp(v_{ax}(t))", "v(t)", "mV", f"Peak-to-peak coordinate range of {ax}")
    # Norm stats (4)
    add_feat("v_norm_mean", ["R0_basic", "R0_rich", "R3", "R4"], "mean(||v(t)||)", "v(t)", "mV", "Mean dipole vector magnitude")
    add_feat("v_norm_max", ["R0_basic", "R0_rich", "R3", "R4"], "max(||v(t)||)", "v(t)", "mV", "Maximum dipole vector magnitude")
    add_feat("v_norm_std", ["R0_basic", "R0_rich", "R3", "R4"], "std(||v(t)||)", "v(t)", "mV", "Standard deviation of dipole magnitude")
    add_feat("v_norm_p95", ["R0_basic", "R0_rich", "R3", "R4"], "percentile(||v(t)||, 95)", "v(t)", "mV", "95th percentile of dipole magnitude")
    # Path length (1)
    add_feat("v_path_length", ["R0_basic", "R0_rich", "R3", "R4"], "sum(||dv||) / T", "v(t)", "mV/sample", "Normalized dipole trajectory spatial path length")
    # Velocity stats (2)
    add_feat("v_velocity_mean", ["R0_basic", "R0_rich", "R3", "R4"], "mean(||dv/dt||)", "v(t)", "mV/s", "Mean dipole velocity")
    add_feat("v_velocity_p95", ["R0_basic", "R0_rich", "R3", "R4"], "percentile(||dv/dt||, 95)", "v(t)", "mV/s", "95th percentile of dipole velocity")
    # Acceleration (1)
    add_feat("v_acceleration_mean", ["R0_basic", "R0_rich", "R3", "R4"], "mean(||d2v/dt2||)", "v(t)", "mV/s^2", "Mean dipole trajectory acceleration")
    # Inertia eigenvalues (3)
    for i in [1, 2, 3]:
        add_feat(f"v_inertia_eig_{i}", ["R0_basic", "R0_rich", "R3", "R4"], f"eigval_{i}(cov(v))", "v(t)", "mV^2", f"Eigenvalue {i} of dipole covariance matrix sorted descending")

    # 2. Rich Physical VCG Controls (78 features: in R0_rich only -> total 23 + 78 = 101)
    # Coordinate Quantiles (15)
    for ax in ["x", "y", "z"]:
        for q in [10, 25, 50, 75, 90]:
            add_feat(f"v_quantile_{ax}_p{q}", ["R0_rich"], f"percentile(v_{ax}(t), {q})", "v(t)", "mV", f"{q}th percentile of coordinate {ax}")
    # Velocity Covariance & Stats (12)
    for pair in ["xx", "xy", "xz", "yy", "yz", "zz"]:
        add_feat(f"v_vel_cov_{pair}", ["R0_rich"], f"cov(vdot_{pair[0]}, vdot_{pair[1]})", "v(t)", "(mV/s)^2", f"Covariance of velocity coordinates {pair}")
    for ax in ["x", "y", "z"]:
        add_feat(f"v_vel_range_{ax}", ["R0_rich"], f"ptp(vdot_{ax}(t))", "v(t)", "mV/s", f"Peak-to-peak velocity range of {ax}")
    add_feat("v_vel_max", ["R0_rich"], "max(||dv/dt||)", "v(t)", "mV/s", "Maximum velocity magnitude")
    add_feat("v_vel_std", ["R0_rich"], "std(||dv/dt||)", "v(t)", "mV/s", "Standard deviation of velocity magnitude")
    add_feat("v_speed_iqr", ["R0_rich"], "iqr(||dv/dt||)", "v(t)", "mV/s", "Interquartile range of velocity speed")
    # Angular Momentum / Loop Dynamics (12)
    for ax in ["x", "y", "z"]:
        add_feat(f"v_ang_mom_mean_{ax}", ["R0_rich"], f"mean((v x vdot)_{ax})", "v(t)", "mV^2/s", f"Mean angular momentum along axis {ax}")
    for pair in ["xx", "xy", "xz", "yy", "yz", "zz"]:
        add_feat(f"v_ang_mom_cov_{pair}", ["R0_rich"], f"cov(L_{pair[0]}, L_{pair[1]})", "v(t)", "(mV^2/s)^2", f"Covariance of angular momentum vector {pair}")
    add_feat("v_ang_mom_mag_mean", ["R0_rich"], "mean(||v x vdot||)", "v(t)", "mV^2/s", "Mean angular momentum loop magnitude")
    add_feat("v_ang_mom_mag_p95", ["R0_rich"], "percentile(||v x vdot||, 95)", "v(t)", "mV^2/s", "95th percentile of angular momentum magnitude")
    add_feat("v_loop_planarity", ["R0_rich"], "eigmin(cov(L)) / sum(eig(cov(L)))", "v(t)", "unitless", "Planar coplanarity ratio of dipole loop")
    # Acceleration Covariance & Multi-scale (12)
    for pair in ["xx", "xy", "xz", "yy", "yz", "zz"]:
        add_feat(f"v_acc_cov_{pair}", ["R0_rich"], f"cov(vddot_{pair[0]}, vddot_{pair[1]})", "v(t)", "(mV/s^2)^2", f"Covariance of acceleration {pair}")
    for ax in ["x", "y", "z"]:
        add_feat(f"v_short_window_var_{ax}", ["R0_rich"], f"mean(rolling_var_50ms(v_{ax}))", "v(t)", "mV^2", f"Mean short-window local variance for coordinate {ax}")
        add_feat(f"v_long_window_var_{ax}", ["R0_rich"], f"mean(rolling_var_200ms(v_{ax}))", "v(t)", "mV^2", f"Mean long-window local variance for coordinate {ax}")
    # Spatial Octant Occupancy & Radial Shells (12)
    for oct_idx in range(8):
        add_feat(f"v_octant_occupancy_{oct_idx}", ["R0_rich"], f"mean(I[octant(v_std(t)) == {oct_idx}])", "v(t)", "fraction", f"Occupancy fraction in spatial octant {oct_idx}")
    for shell_idx, shell_label in enumerate(["0_to_0.5", "0.5_to_1.0", "1.0_to_2.0", "gt_2.0"]):
        add_feat(f"v_radial_shell_{shell_label}", ["R0_rich"], f"mean(I[||v_std(t)|| in {shell_label}])", "v(t)", "fraction", f"Occupancy fraction in standardized radial shell {shell_label}")
    # Phase-Plane Bivariate Statistics (9)
    for ax in ["x", "y", "z"]:
        add_feat(f"v_phase_corr_{ax}", ["R0_rich"], f"corr(v_{ax}(t), vdot_{ax}(t))", "v(t)", "unitless", f"Correlation between coordinate {ax} and its velocity")
        add_feat(f"v_phase_area_{ax}", ["R0_rich"], f"ptp(v_{ax}) * ptp(vdot_{ax})", "v(t)", "mV^2/s", f"Phase-plane bounding box area for coordinate {ax}")
        add_feat(f"v_vel_skew_{ax}", ["R0_rich"], f"skew(vdot_{ax}(t))", "v(t)", "unitless", f"Skewness of velocity coordinate {ax}")
    # Temporal Dynamics (6)
    for ax in ["x", "y", "z"]:
        add_feat(f"v_temporal_mean_drift_{ax}", ["R0_rich"], f"mean(v_{ax}[T/2:]) - mean(v_{ax}[:T/2])", "v(t)", "mV", f"Half-split mean drift for coordinate {ax}")
    add_feat("v_temporal_energy_drift", ["R0_rich"], "mean(||v[T/2:]||^2) - mean(||v[:T/2]||^2)", "v(t)", "mV^2", "Half-split energy drift")
    add_feat("v_norm_autocorr_lag1", ["R0_rich"], "autocorr(||v(t)||, lag=50samples)", "v(t)", "unitless", "Autocorrelation of dipole magnitude at 100ms lag")
    add_feat("v_norm_autocorr_lag2", ["R0_rich"], "autocorr(||v(t)||, lag=100samples)", "v(t)", "unitless", "Autocorrelation of dipole magnitude at 200ms lag")

    # 3. Spatial Domain Occupancy Features (67 features: in R1 only)
    for d in range(64):
        add_feat(f"domain_occupancy_{d:02d}", ["R1"], f"mean(I[G(v(t)) == {d}])", "G(v(t))", "fraction", f"Temporal occupancy fraction of spatial domain {d}")
    add_feat("domain_occupancy_entropy", ["R1"], "-sum(p_g * log(p_g + eps))", "G(v(t))", "nats", "Shannon entropy of spatial domain occupancy distribution")
    add_feat("domain_unique_count", ["R1"], "sum(I[p_g > 0])", "G(v(t))", "count", "Number of distinct spatial domains visited")
    add_feat("domain_mean_dwell", ["R1"], "T / max(n_transitions, 1)", "G(v(t))", "samples", "Mean dwell time in samples per spatial domain visit")

    # 4. Hilbert Functional Trajectory Features (68 features: in R2_H, R3, R4)
    for h in range(9):
        add_feat(f"h_mean_{h:02d}", ["R2_H", "R3", "R4"], f"mean(h_H,{h}(t))", "h_H(t)", "RKHS_units", f"Temporal mean of Hilbert coordinate {h}")
    for i in range(9):
        for j in range(i, 9):
            add_feat(f"h_cov_{i:02d}_{j:02d}", ["R2_H", "R3", "R4"], f"cov(h_H,{i}, h_H,{j})", "h_H(t)", "RKHS_units^2", f"Covariance of Hilbert coordinates {i} and {j}")
    for h in range(9):
        add_feat(f"h_range_{h:02d}", ["R2_H", "R3", "R4"], f"ptp(h_H,{h}(t))", "h_H(t)", "RKHS_units", f"Peak-to-peak coordinate range of Hilbert axis {h}")
    add_feat("h_path_length", ["R2_H", "R3", "R4"], "sum(||dh_H||) / T", "h_H(t)", "RKHS_units/sample", "Normalized path length in Hilbert RKHS space")
    add_feat("h_trans_dist_mean", ["R2_H", "R3", "R4"], "mean(||dh_H[transitions]||)", "h_H(t)", "RKHS_units", "Mean jump distance in Hilbert space during domain transitions")
    add_feat("h_dwell_mean", ["R2_H", "R3", "R4"], "T / max(n_transitions, 1)", "G(v(t))", "samples", "Mean dwell time per functional microstate domain")
    add_feat("h_occ_entropy", ["R2_H", "R3", "R4"], "-sum(p_g * log(p_g + eps))", "G(v(t))", "nats", "Functional domain occupancy entropy")
    add_feat("h_norm_mean", ["R2_H", "R3", "R4"], "mean(||h_H(t)||)", "h_H(t)", "RKHS_units", "Mean magnitude of Hilbert coordinate vector")

    # 5. Physical-Functional Coupling Features (10 features: in R3, R4)
    for vi, v_ax in enumerate(["x", "y", "z"]):
        for hj in range(3):
            add_feat(f"vh_cross_cov_{v_ax}_h{hj}", ["R3", "R4"], f"cov(v_{v_ax}, h_H,{hj})", "v(t), h_H(t)", "mV*RKHS_units", f"Cross-covariance between physical {v_ax} and Hilbert coordinate {hj}")
    add_feat("vh_norm_corr", ["R3", "R4"], "corr(||v(t)||, ||h_H(t)||)", "v(t), h_H(t)", "unitless", "Correlation between physical dipole magnitude and Hilbert functional magnitude")

    # 6. Non-Dipolar Residual Features (21 features: in R4 only -> total 101 + 21 = 122)
    lead_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    for l_name in lead_names:
        add_feat(f"residual_rms_{l_name}", ["R4"], f"sqrt(mean(r_{l_name}(t)^2))", "r(t)", "mV", f"Root Mean Square residual error on lead {l_name}")
    add_feat("residual_precordial_ratio", ["R4"], "sum(rms(r_V1..V6)^2) / sum(rms(r_I,II)^2 + eps)", "r(t)", "ratio", "Precordial to limb non-dipolar residual power ratio")
    for l_name in lead_names:
        add_feat(f"residual_fraction_{l_name}", ["R4"], f"rms(r_{l_name}) / rms(E_{l_name} + eps)", "r(t), E(t)", "fraction", f"Fraction of lead {l_name} signal energy unexplained by 3D dipole")
    for l_name in ["I", "II", "V2", "V5"]:
        add_feat(f"residual_hf_power_{l_name}", ["R4"], f"mean((dr_{l_name}/dt)^2)", "r(t)", "(mV/s)^2", f"High-frequency residual noise/local activity on lead {l_name}")

    return schema


def main():
    schema = build_schema()
    
    # Audit dimension counts
    counts = {"R0_basic": 0, "R0_rich": 0, "R1": 0, "R2_H": 0, "R3": 0, "R4": 0}
    for f in schema["features"]:
        for b in f["baseline_memberships"]:
            counts[b] += 1
            
    print("Schema Baseline Dimension Audit:")
    for b, count in counts.items():
        expected = schema["baselines_summary"][b]["dimension"]
        status = "OK" if count == expected else "MISMATCH"
        print(f"  - {b:10s}: {count:3d} features (expected: {expected:3d}) -> {status}")
        assert count == expected, f"Dimension mismatch for {b}: got {count}, expected {expected}"

    out_path = Path("refine-logs/qvcg/reference_representation/REFERENCE_REPRESENTATION_SCHEMA.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"\nSaved canonical schema with {len(schema['features'])} total feature definitions to {out_path}")


if __name__ == "__main__":
    main()
