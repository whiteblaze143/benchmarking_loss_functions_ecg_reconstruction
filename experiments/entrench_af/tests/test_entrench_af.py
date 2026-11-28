"""ENTRENCH-AF Acceptance Test Suite (7 required tests).

Run with:
    /home/mithunmanivannan/.venv/bin/pytest -v -s experiments/entrench_af/tests/test_entrench_af.py

Set ENTRENCH_AF_FINAL=1 for the final end-to-end verification so that any
missing stage output fails instead of skipping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/entrench_af")
CONFIG_DIR = ROOT / "config"
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
FINAL_MODE = os.environ.get("ENTRENCH_AF_FINAL", "0") == "1"


def _require_path(path: Path, label: str) -> Path:
    if not path.exists():
        if FINAL_MODE:
            pytest.fail(f"[FINAL MODE] Missing required artifact ({label}): {path}")
        pytest.skip(f"Pending milestone artifact ({label}): {path}")
    return path


def _latest_run_dir() -> Path:
    pointer = _require_path(RESULTS_DIR / "LATEST_RUN.txt", "LATEST_RUN.txt")
    raw = pointer.read_text().strip()
    run_dir = Path(raw)
    if not run_dir.is_absolute():
        candidate = RESULTS_DIR / raw
        run_dir = candidate if candidate.exists() else (ROOT / raw)
    return _require_path(run_dir, f"run directory {run_dir}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_module(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None, f"Cannot load {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: Inventory hashes unchanged after Stage 0
# ---------------------------------------------------------------------------
def test_01_inventory_frozen() -> None:
    run_dir = _latest_run_dir()
    inv_path = _require_path(run_dir / "stage0" / "inventory_frozen.csv", "Stage 0 inventory")
    def_hash_path = _require_path(run_dir / "stage0" / "definitions_hash.txt", "Stage 0 definitions_hash.txt")
    def_yaml_path = _require_path(CONFIG_DIR / "definitions.yaml", "config/definitions.yaml")

    expected_def_hash = _sha256_file(def_yaml_path)
    recorded_def_hash = def_hash_path.read_text().strip().split()[0]
    assert recorded_def_hash == expected_def_hash, (
        f"definitions_hash.txt mismatch: recorded={recorded_def_hash} vs actual={expected_def_hash}"
    )

    df = pd.read_csv(inv_path)
    assert len(df) == 379, f"Expected 379 total records, got {len(df)}"

    # Identify patient column
    pt_col = next((c for c in ["patient", "patient_id", "subject_id"] if c in df.columns), None)
    assert pt_col is not None, f"No patient column found in {list(df.columns)}"
    assert df[pt_col].nunique() == 358, f"Expected 358 unique patients, got {df[pt_col].nunique()}"

    # Annotated subset
    ann_col = next((c for c in ["is_annotated", "annotated", "has_annotations"] if c in df.columns), None)
    assert ann_col is not None, f"No annotation flag column found in {list(df.columns)}"
    ann_mask = df[ann_col].astype(bool)
    ann_df = df[ann_mask]
    assert len(ann_df) == 349, f"Expected 349 annotated records, got {len(ann_df)}"
    assert ann_df[pt_col].nunique() == 329, f"Expected 329 annotated patients, got {ann_df[pt_col].nunique()}"

    # 30 unannotated SHDB records
    unann_df = df[~ann_mask]
    assert len(unann_df) == 30, f"Expected 30 unannotated records, got {len(unann_df)}"
    cohort_col = next((c for c in ["cohort", "dataset"] if c in df.columns), None)
    if cohort_col is not None:
        assert set(unann_df[cohort_col].astype(str).str.upper().unique()) == {"SHDB"}, (
            f"Unannotated records should all be SHDB, got {unann_df[cohort_col].unique()}"
        )

    # Raw runs count == 8,540 (from inventory column or stage0_summary.json)
    runs_col = next((c for c in ["n_raw_runs", "raw_runs", "n_af_runs", "af_runs", "n_runs"] if c in df.columns), None)
    if runs_col is not None:
        total_raw_runs = int(df[runs_col].fillna(0).sum())
    else:
        summary_path = _require_path(run_dir / "stage0" / "stage0_summary.json", "Stage 0 summary")
        summary = json.loads(summary_path.read_text())
        total_raw_runs = int(summary["n_runs_pre_harmonisation"])
    assert total_raw_runs == 8540, f"Expected 8,540 raw runs, got {total_raw_runs}"

    # Sample discrepancy == 0
    disc_col = next((c for c in ["sample_discrepancy", "discrepancy_samples", "discrepancy"] if c in df.columns), None)
    assert disc_col is not None, f"No sample discrepancy column found in {list(df.columns)}"
    max_disc = df[disc_col].fillna(0).abs().max()
    assert max_disc == 0, f"Expected sample discrepancy == 0, got max={max_disc}"

    # SHA-256 hash present for every record
    hash_col = next(
        (c for c in ["primary_file_sha256", "file_sha256", "sha256", "record_sha256"] if c in df.columns),
        None,
    )
    assert hash_col is not None, f"No sha256 column found in {list(df.columns)}"
    assert df[hash_col].notna().all(), "Some records are missing sha256 hashes"
    assert (df[hash_col].astype(str).str.len() == 64).all(), "Invalid sha256 hash length"


# ---------------------------------------------------------------------------
# Test 2: Burden invariant across gap-merge family & monotone run counts
# ---------------------------------------------------------------------------
def test_02_burden_invariant_and_monotone_runs() -> None:
    run_dir = _latest_run_dir()
    stage2_dir = _require_path(run_dir / "stage2", "Stage 2 directory")

    gaps = [0, 30, 120, 360, 2400]
    counts: dict[int, int] = {}
    burdens: dict[int, float] = {}

    for g in gaps:
        p = _require_path(stage2_dir / f"runs_g{g}.parquet", f"runs_g{g}.parquet")
        df = pd.read_parquet(p)
        counts[g] = len(df)
        assert "dur_sec" in df.columns, f"Missing dur_sec in runs_g{g}.parquet"
        assert (df["dur_sec"] >= 0).all(), f"Negative dur_sec in runs_g{g}.parquet"
        # AF burden: if af_only_dur_sec / af_dur_sec exists (excluding merged non-AF gaps), use it; else dur_sec
        burden_col = next((c for c in ["af_only_dur_sec", "af_dur_sec", "dur_sec"] if c in df.columns), "dur_sec")
        burdens[g] = float(df[burden_col].sum())

    # Monotone non-increasing run counts
    for g1, g2 in zip(gaps, gaps[1:]):
        assert counts[g1] >= counts[g2], (
            f"Run counts not monotone non-increasing: g={g1} has {counts[g1]} < g={g2} has {counts[g2]}"
        )

    # Burden invariant within +/- 0.5% relative to g0
    base_burden = burdens[0]
    assert base_burden > 0, "Base burden at g0 must be positive"
    for g in gaps[1:]:
        rel_diff = abs(burdens[g] - base_burden) / base_burden
        assert rel_diff <= 0.005, (
            f"Burden at g={g} ({burdens[g]:.1f}s) deviates from g0 ({base_burden:.1f}s) "
            f"by {rel_diff * 100:.3f}% (> 0.5%)"
        )
    print(f"\n[Test 2] Run counts: {counts}")
    print(f"[Test 2] Burdens (h): { {g: round(b / 3600.0, 2) for g, b in burdens.items()} }")


# ---------------------------------------------------------------------------
# Test 3: Event counts monotone across min-run filters & zero negative times
# ---------------------------------------------------------------------------
def test_03_event_counts_monotone_and_nonnegative_reinit() -> None:
    run_dir = _latest_run_dir()
    stage4_dir = _require_path(run_dir / "stage4", "Stage 4 directory")

    ev0_path = _require_path(stage4_dir / "events_g0.parquet", "events_g0.parquet")
    ev0 = pd.read_parquet(ev0_path)
    assert "time_to_reinit_sec" in ev0.columns, f"Missing time_to_reinit_sec in {list(ev0.columns)}"
    assert (ev0["time_to_reinit_sec"].dropna() >= 0).all(), (
        f"Found negative time_to_reinit_sec in events_g0.parquet: "
        f"min={ev0['time_to_reinit_sec'].min()}"
    )

    # Check sensitivity min-run filters (30, 120, 300 s)
    # Either separate files (events_g0_min120.parquet / events_min120.parquet) or derived from dur_sec
    counts = {30: len(ev0)}
    for min_s in [120, 300]:
        candidates = [
            stage4_dir / f"events_g0_min{min_s}.parquet",
            stage4_dir / f"events_min{min_s}.parquet",
            stage4_dir / f"events_g0_minrun{min_s}.parquet",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is not None:
            df_s = pd.read_parquet(found)
            assert (df_s["time_to_reinit_sec"].dropna() >= 0).all(), f"Negative time_to_reinit_sec in {found.name}"
            counts[min_s] = len(df_s)
        else:
            dur_col = next((c for c in ["dur_sec", "run_dur_sec"] if c in ev0.columns), None)
            assert dur_col is not None, f"No sensitivity file for min={min_s}s and no dur_sec column"
            counts[min_s] = int((ev0[dur_col] >= min_s).sum())

    assert counts[30] >= counts[120] >= counts[300], f"Event counts not monotone: {counts}"
    print(f"\n[Test 3] Event counts across min-run filters: {counts}")


# ---------------------------------------------------------------------------
# Test 4: Reinitiation proportions at g0 (<360 s) vs inventory values
# ---------------------------------------------------------------------------
def test_04_reinitiation_proportions_g0() -> None:
    run_dir = _latest_run_dir()
    ev0_path = _require_path(run_dir / "stage4" / "events_g0.parquet", "events_g0.parquet")
    df = pd.read_parquet(ev0_path)

    cohort_col = next((c for c in ["cohort", "dataset"] if c in df.columns), None)
    assert cohort_col is not None, f"No cohort column in {list(df.columns)}"

    # Find <360s reinitiation binary column or compute from time_to_reinit_sec + event
    reinit_col = next(
        (c for c in ["reinit_360", "reinit_360s", "reinit_lt_360", "reinit_within_360s"] if c in df.columns),
        None,
    )
    if reinit_col is not None:
        reinit_360 = df[reinit_col].astype(bool)
    else:
        ev_col = next((c for c in ["event", "reinit_event"] if c in df.columns), None)
        assert ev_col is not None, f"Neither reinit_360 nor event column found in {list(df.columns)}"
        reinit_360 = df[ev_col].astype(bool) & (df["time_to_reinit_sec"] < 360.0)

    expected = {"IRIDIA": 0.18, "SHDB": 0.72, "LTAFDB": 0.95}
    observed: dict[str, float] = {}
    shifted: list[str] = []
    for cohort, target in expected.items():
        mask = df[cohort_col].astype(str).str.upper() == cohort
        assert mask.sum() > 0, f"No events found for cohort {cohort}"
        prop = float(reinit_360[mask].mean())
        observed[cohort] = prop
        diff = abs(prop - target)
        if diff > 0.05:
            shifted.append(f"{cohort}: obs={prop:.4f} vs inv={target:.2f} (delta={prop - target:+.4f})")

    print(f"\n[Test 4] g0 reinit <360s proportions: { {k: round(v, 4) for k, v in observed.items()} }")
    if shifted:
        print(
            "[Test 4] FLAG: Harmonized 5-beat rule shifted reinitiation proportions >5pp: "
            + "; ".join(shifted)
        )
    # Sanity ordering: IRIDIA < SHDB < LTAFDB must still hold
    assert observed["IRIDIA"] < observed["SHDB"] < observed["LTAFDB"], (
        f"Unexpected cohort ordering for reinit <360s: {observed}"
    )


# ---------------------------------------------------------------------------
# Test 5: Aim 2 Spitaler emulation unit test on synthetic day-level fixture
# ---------------------------------------------------------------------------
def test_05_aim2_spitaler_emulation_unit() -> None:
    script_path = _require_path(SCRIPTS_DIR / "08_aim2_reconstruction_equivalence.py", "Aim 2 script")
    mod = _load_module(script_path, "aim2_recon")
    assert hasattr(mod, "emulate_spitaler_episodes"), (
        "08_aim2_reconstruction_equivalence.py must export emulate_spitaler_episodes(runs_df, rules)"
    )

    rules = {
        "min_daily_burden_min": 6,
        "full_day_min": 1400,
        "intra_day_tolerance_min": 40,
        "bridge_midnight_requires_full_day": True,
    }

    DAY = 86400.0
    # Construct a 12-day synthetic recording (days 0..11) for 1 record:
    # Day 0: 600s (10 min) -> boundary day (should be excluded!)
    # Day 1: 0s (gap)
    # Day 2: 600s (10 min, partial)
    # Day 3: 85000s (1416.7 min >= 1400, FULL)
    # Day 4: 1200s (20 min, partial)
    #   -> Days 2-3-4: partial-full-partial = ONE episode (dur = 600 + 85000 + 1200 = 86800s)
    # Day 5: 0s (gap breaks!)
    # Day 6: 600s (10 min, partial)
    # Day 7: 900s (15 min, partial)
    #   -> Days 6-7: partial-partial = TWO separate episodes (600s and 900s)
    # Day 8: 0s (gap)
    # Day 9: 300s (5 min < 6 min) -> filtered out!
    # Day 10: 0s (gap)
    # Day 11: 1200s (20 min) -> boundary day (should be excluded!)
    rows = [
        # (day_idx, dur_sec)
        (0, 600.0),
        (2, 600.0),
        (3, 85000.0),
        (4, 1200.0),
        (6, 600.0),
        (7, 900.0),
        (9, 300.0),
        (11, 1200.0),
    ]
    records = []
    for idx, (d, dur) in enumerate(rows):
        onset = d * DAY + 100.0
        records.append(
            {
                "record": "SYNTH_01",
                "patient": "PT_SYNTH_01",
                "cohort": "IRIDIA",
                "run_id": idx,
                "onset_sec": onset,
                "offset_sec": onset + dur,
                "dur_sec": dur,
                "record_dur_sec": 12.0 * DAY,
                "boundary_censored_start": (d == 0),
                "boundary_censored_end": (d == 11),
            }
        )
    runs_df = pd.DataFrame(records)
    out = mod.emulate_spitaler_episodes(runs_df, rules)
    assert isinstance(out, pd.DataFrame), "emulate_spitaler_episodes must return a DataFrame"

    # Filter out boundary episodes if the function returns them with a flag instead of dropping them
    for bcol in ["is_boundary", "boundary_excluded", "boundary_censored"]:
        if bcol in out.columns:
            out = out[~out[bcol].astype(bool)].copy()

    dur_col = next((c for c in ["dur_sec", "episode_dur_sec", "af_dur_sec"] if c in out.columns), None)
    assert dur_col is not None, f"No duration column in emulated episodes: {list(out.columns)}"

    durs = sorted(out[dur_col].round(1).tolist())
    expected_durs = sorted([86800.0, 600.0, 900.0])
    assert len(durs) == 3, f"Expected 3 non-boundary episodes (P-F-P + P + P), got {len(durs)}: {durs}"
    for obs, exp in zip(durs, expected_durs):
        assert abs(obs - exp) < 1.0, f"Episode duration mismatch: observed={durs}, expected={expected_durs}"
    print(f"\n[Test 5] Synthetic Spitaler emulation passed: durations={durs}")


# ---------------------------------------------------------------------------
# Test 6: Weibull frailty fit converges; shape CI logged
# ---------------------------------------------------------------------------
def test_06_weibull_frailty_convergence() -> None:
    run_dir = _latest_run_dir()
    p = _require_path(run_dir / "aim1" / "tableS1_weibull_frailty.csv", "Aim 1 Weibull frailty table")
    df = pd.read_csv(p)
    assert len(df) > 0, "tableS1_weibull_frailty.csv is empty"

    shape_col = next((c for c in ["shape_k", "shape", "weibull_shape", "k", "gamma_shape"] if c in df.columns), None)
    assert shape_col is not None, f"No shape column found in {list(df.columns)}"
    assert df[shape_col].notna().all(), "NaN Weibull shape values found"
    assert (df[shape_col] > 0).all(), "Non-positive Weibull shape values found"

    if "converged" in df.columns:
        assert df["converged"].astype(bool).all(), f"Non-converged Weibull frailty fits:\n{df[~df['converged']]}"

    ci_lo = next((c for c in ["shape_ci_low", "ci_low", "shape_lo", "ci_lower"] if c in df.columns), None)
    ci_hi = next((c for c in ["shape_ci_high", "ci_high", "shape_hi", "ci_upper"] if c in df.columns), None)
    assert ci_lo is not None and ci_hi is not None, f"Missing shape CI columns in {list(df.columns)}"

    for _, row in df.iterrows():
        lo, hi, sh = float(row[ci_lo]), float(row[ci_hi]), float(row[shape_col])
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= sh <= hi, (
            f"Invalid shape CI: {lo} <= {sh} <= {hi}"
        )
        excludes_1 = (hi < 1.0) or (lo > 1.0)
        label = row.get("cohort", row.get("strata", "row"))
        print(f"\n[Test 6] {label}: shape={sh:.4f} [{lo:.4f}, {hi:.4f}] (excludes 1: {excludes_1})")


# ---------------------------------------------------------------------------
# Test 7: No result file overwritten / every output CSV carries provenance
# ---------------------------------------------------------------------------
def test_07_provenance_and_no_overwrite() -> None:
    run_dir = _latest_run_dir()
    assert run_dir.name.startswith("run_"), (
        f"Expected timestamped run_<ts>_<hash> directory, got {run_dir.name}"
    )

    csv_files = sorted(run_dir.rglob("*.csv"))
    assert len(csv_files) > 0, f"No output CSV files found under {run_dir}"

    required_cols = {"git_hash", "definitions_hash", "run_timestamp"}
    missing_report: list[str] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, nrows=5)
        missing = required_cols - set(df.columns)
        if missing:
            missing_report.append(f"{csv_path.relative_to(run_dir)}: missing {sorted(missing)}")
        elif len(df) > 0:
            for col in required_cols:
                if df[col].isna().any():
                    missing_report.append(f"{csv_path.relative_to(run_dir)}: NaN in {col}")

    assert not missing_report, "Provenance check failed:\n  " + "\n  ".join(missing_report)
    print(f"\n[Test 7] Verified provenance columns on {len(csv_files)} CSV files in {run_dir.name}")


# ---------------------------------------------------------------------------
# Test 8: Reference DOIs verified & banned DOIs absent
# ---------------------------------------------------------------------------
def test_08_reference_dois_verified() -> None:
    report_path = _require_path(ROOT / "reports" / "ENTRENCH_AF_Report.md", "ENTRENCH_AF_Report.md")
    refs_yaml_path = _require_path(CONFIG_DIR / "references.yaml", "config/references.yaml")
    verify_script = _require_path(SCRIPTS_DIR / "verify_references.py", "scripts/verify_references.py")

    mod = _load_module(verify_script, "verify_references")
    assert hasattr(mod, "verify_markdown_references"), (
        "verify_references.py must export verify_markdown_references(md_path, refs_yaml_path)"
    )
    mod.verify_markdown_references(report_path, refs_yaml_path)

    report_text = report_path.read_text(encoding="utf-8")
    required_dois = [
        "10.1016/j.hrthm.2026.10.019",   # Spitaler 2026
        "10.1038/s41597-023-02621-1",    # Gilon 2023 (IRIDIA-AF)
        "10.1038/s41597-025-04777-4",    # Tsutsui 2025 (SHDB-AF)
    ]
    for doi in required_dois:
        assert doi.lower() in report_text.lower(), f"Missing required DOI {doi} in ENTRENCH_AF_Report.md"

    banned_dois = [
        "10.1093/europace/euad122",
        "10.1038/s41597-023-02632-y",
        "10.13026/shdb-af",
    ]
    hits: list[str] = []
    for search_dir in [ROOT / "reports", SCRIPTS_DIR]:
        if not search_dir.exists():
            continue
        for p in sorted(search_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() in {".png", ".pdf", ".parquet", ".pyc"}:
                continue
            if p.name == "verify_references.py":
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            for b in banned_dois:
                if b.lower() in txt:
                    hits.append(f"{p.relative_to(ROOT)}: contains banned DOI {b}")
    assert not hits, "Banned DOIs found:\n  " + "\n  ".join(hits)
    print("\n[Test 8] Reference DOIs verified and banned DOIs absent across reports/ and scripts/")


# ---------------------------------------------------------------------------
# Test 9: Aim 1 hazard profile, Spitaler grid, S10 side-by-side, S2 dropped
# ---------------------------------------------------------------------------
def test_09_aim1_hazard_spitaler_and_minrun() -> None:
    run_dir = _latest_run_dir()

    # 1. Table S2 dropped
    assert not (run_dir / "aim1" / "tableS2_changepoint_lrt.csv").exists(), (
        "tableS2_changepoint_lrt.csv should be dropped in Aim 1"
    )
    assert not (ROOT / "reports" / "tables" / "S2_changepoint_lrt.csv").exists(), (
        "reports/tables/S2_changepoint_lrt.csv should be removed"
    )

    # 2. Output #2: hazard_profile.csv
    hp_path = _require_path(run_dir / "aim1" / "hazard_profile.csv", "aim1/hazard_profile.csv")
    hp = pd.read_csv(hp_path)
    assert len(hp) > 0, "hazard_profile.csv is empty"
    assert "suppressed_lt20" in hp.columns, f"Missing suppressed_lt20 in {list(hp.columns)}"
    if "estimator" in hp.columns:
        ests = set(hp["estimator"].astype(str).unique())
        assert "nelson_aalen_kernel" in ests, f"Missing nelson_aalen_kernel in {ests}"

    # 3. Output #3: table1_spitaler_grid.csv with <20 suppression & FULL published Spitaler Table 3 values (zero NaNs)
    sg_path = _require_path(run_dir / "aim1" / "table1_spitaler_grid.csv", "aim1/table1_spitaler_grid.csv")
    sg = pd.read_csv(sg_path)
    for col in ["p_term_km", "suppressed_lt20", "spitaler_published_p", "spitaler_ci_low", "spitaler_ci_high", "threshold_h"]:
        assert col in sg.columns, f"Missing {col} in {list(sg.columns)}"
    assert sg.isna().sum().sum() == 0, f"table1_spitaler_grid.csv must contain zero NaNs, found:\n{sg.isna().sum()}"
    expected_sp = {1: 0.911, 2: 0.896, 3: 0.882, 4: 0.867, 5: 0.850, 6: 0.834, 12: 0.721, 24: 0.333, 48: 0.216}
    for th_h, val in expected_sp.items():
        got = sg[sg["threshold_h"] == th_h]["spitaler_published_p"].unique()
        assert len(got) == 1 and abs(float(got[0]) - val) < 1e-4, f"Expected Spitaler {th_h}h={val}, got {got}"
    sp_hp = hp[hp["cohort"] == "SPITALER_REF"]
    assert len(sp_hp) == 14, f"Expected all 14 Spitaler Table 3 rows (1h..168h) in hazard_profile.csv, got {len(sp_hp)}"
    assert sp_hp[["hazard_per_hour", "hazard_ci_lo", "hazard_ci_hi", "n_at_risk_start", "n_events"]].notna().all().all(), (
        "SPITALER_REF rows in hazard_profile.csv must have zero NaNs"
    )

    # 4. Output #4: S10 side-by-side weibull_shape_vs_min_run.csv + S4 figure
    minrun_csv = _require_path(run_dir / "aim1" / "weibull_shape_vs_min_run.csv", "weibull_shape_vs_min_run.csv")
    mr = pd.read_csv(minrun_csv)
    assert len(mr) >= 8, f"Expected >=8 rows in weibull_shape_vs_min_run.csv, got {len(mr)}"
    for col in ["shape_k_left_trunc", "shape_k_shifted_origin"]:
        assert col in mr.columns, f"Missing side-by-side column {col} in {list(mr.columns)}"
    s4_fig = _require_path(ROOT / "reports" / "figures" / "S4_weibull_shape_vs_min_run.png", "S4 figure")
    assert s4_fig.stat().st_size > 1000, "S4_weibull_shape_vs_min_run.png appears empty"
    print("\n[Test 9] Aim 1 hazard_profile, FULL Spitaler Table 3 grid (zero NaNs), S10 side-by-side & S2 drop verified")


# ---------------------------------------------------------------------------
# Test 10: Aim 3 monotone landmark flow, zero imputation, and HRS abstract honesty
# ---------------------------------------------------------------------------
def test_10_aim3_clean_refit_and_abstract_honesty() -> None:
    run_dir = _latest_run_dir()

    elig_path = _require_path(run_dir / "aim3" / "eligibility_flow.csv", "aim3/eligibility_flow.csv")
    vif_path = _require_path(run_dir / "aim3" / "collinearity_vif.csv", "aim3/collinearity_vif.csv")
    elig = pd.read_csv(elig_path)
    assert len(elig) > 0, "eligibility_flow.csv is empty"
    assert len(pd.read_csv(vif_path)) > 0, "collinearity_vif.csv is empty"

    # Strictly monotone nested gate flow for POOLED
    pool_elig = elig[elig["cohort"] == "POOLED"]["n_events"].to_list()
    assert all(a >= b for a, b in zip(pool_elig, pool_elig[1:])), (
        f"POOLED eligibility_flow is not monotone non-increasing: {pool_elig}"
    )

    t3_path = _require_path(run_dir / "aim3" / "table3_primary_hrs.csv", "aim3/table3_primary_hrs.csv")
    t3 = pd.read_csv(t3_path)
    assert "aft_time_ratio" not in t3.columns, "Broken unstratified aft_time_ratio column must be removed"

    model_col = next((c for c in ["model", "model_spec", "spec", "analysis"] if c in t3.columns), None)
    assert model_col is not None, f"No model spec column in {list(t3.columns)}"
    assert "legacy_unfiltered_imputed_n2346" not in set(t3[model_col].astype(str)), "Imputed model must not exist"
    prim = t3[t3[model_col].astype(str) == "primary_marker_eligible_complete_case"]
    assert len(prim) > 0, f"primary_marker_eligible_complete_case not found in {t3[model_col].unique()}"
    n_ev_col = next((c for c in ["n_events", "n", "N_events"] if c in t3.columns), None)
    assert n_ev_col is not None, f"No n_events column in {list(t3.columns)}"
    assert (prim[n_ev_col] < 600).all(), (
        f"primary_marker_eligible_complete_case must have n_events < 600, got {prim[n_ev_col].unique()}"
    )

    assert "h4b_attenuation_ratio" in t3.columns, f"Missing h4b_attenuation_ratio in {list(t3.columns)}"
    h4b_vals = pd.to_numeric(t3["h4b_attenuation_ratio"], errors="coerce").dropna()
    assert len(h4b_vals) > 0 and np.isfinite(h4b_vals).all(), "h4b_attenuation_ratio has no finite values"

    ts_path = _require_path(run_dir / "aim3" / "two_state_summary.csv", "aim3/two_state_summary.csv")
    ts = pd.read_csv(ts_path)
    assert "sr_reinit_frailty_shape_k" in ts.columns, (
        f"Missing sr_reinit_frailty_shape_k in {list(ts.columns)}"
    )

    # Matched LOCO validation must include history_only_matched_449 baseline
    ts3 = pd.read_csv(_require_path(run_dir / "aim3" / "tableS3_loco_validation.csv", "tableS3_loco_validation.csv"))
    assert "history_only_matched_449" in set(ts3["model_spec"].astype(str)), "Missing history_only_matched_449 validation baseline"

    # Focused HRS 2027 Abstract + Claim Registry checks
    hrs_path = _require_path(ROOT / "reports" / "HRS_2027_Abstract.md", "HRS_2027_Abstract.md")
    claim_path = _require_path(ROOT / "reports" / "CLAIM_REGISTRY.md", "CLAIM_REGISTRY.md")
    hrs_text = hrs_path.read_text(encoding="utf-8")
    assert "Background:" in hrs_text and "Objective:" in hrs_text and "Methods:" in hrs_text and "Results:" in hrs_text and "Conclusion:" in hrs_text
    assert "consistent discrimination across" not in hrs_text.lower()
    assert "Spitaler" in hrs_text and "Doehner" in hrs_text and "Temporal Aggregation Alters" in hrs_text
    claim_text = claim_path.read_text(encoding="utf-8")
    assert "Established" in claim_text and "Exploratory" in claim_text and "Unresolved Limitations" in claim_text and "Wineinger" in claim_text
    print(f"\n[Test 10] Aim 3 clean refit (n_events={prim[n_ev_col].iloc[0]}), monotone flow, & HRS abstract verified")


# ---------------------------------------------------------------------------
# Test 11: Aim 4 & Secondary fixes + prior run untouched
# ---------------------------------------------------------------------------
def test_11_aim4_and_secondary_fixes() -> None:
    run_dir = _latest_run_dir()

    # Prior run directory must remain untouched and distinct from LATEST_RUN
    prior_run = RESULTS_DIR / "run_20261126T193111Z_cc84d8b0"
    assert prior_run.exists(), "Prior run directory run_20261126T193111Z_cc84d8b0 is missing"
    assert run_dir.name != prior_run.name, (
        f"LATEST_RUN.txt must point to a fresh revision run dir, still points to {prior_run.name}"
    )

    # Aim 4 checks
    t4_path = _require_path(run_dir / "aim4" / "table4_duration_distributions.csv", "Aim 4 table")
    t4 = pd.read_csv(t4_path)
    if "xmin_mode" in t4.columns:
        ff = t4[t4["xmin_mode"].astype(str) == "fixed_filter"]
    else:
        ff = t4
    iridia_ff = ff[ff["cohort"].astype(str).str.upper() == "IRIDIA"]
    filter_col = next((c for c in ["filter", "subset", "min_run_filter"] if c in t4.columns), None)
    assert filter_col is not None, f"No filter column in {list(t4.columns)}"
    x30 = iridia_ff[iridia_ff[filter_col].astype(str) == "ge_30s"]["xmin_sec"].tolist()
    x360 = iridia_ff[iridia_ff[filter_col].astype(str) == "ge_360s"]["xmin_sec"].tolist()
    assert len(x30) > 0 and len(x360) > 0, f"Missing IRIDIA ge_30s/ge_360s in fixed_filter: {x30}, {x360}"
    assert abs(float(x30[0]) - 30.0) < 1e-6 and abs(float(x360[0]) - 360.0) < 1e-6, (
        f"Expected distinct xmin_sec 30 vs 360 for IRIDIA fixed_filter, got {x30[0]} vs {x360[0]}"
    )

    if "n_tail" in t4.columns and "best_aic_model" in t4.columns:
        small_tail = t4[t4["n_tail"] < 50]
        if len(small_tail) > 0:
            bad = small_tail[small_tail["best_aic_model"].astype(str) != "insufficient_tail_n<50"]
            assert len(bad) == 0, f"Small-tail (<50) rows did not suppress best_aic_model:\n{bad}"

    # Secondary checks
    sub_path = _require_path(run_dir / "secondary" / "shdb_subgroups.csv", "secondary/shdb_subgroups.csv")
    sub = pd.read_csv(sub_path)
    assert "p_reinit_60s_between_groups" in sub.columns, f"Missing p_reinit_60s_between_groups in {list(sub.columns)}"
    p60 = pd.to_numeric(sub["p_reinit_60s_between_groups"], errors="coerce").dropna()
    assert len(p60) > 0 and np.isfinite(p60).all(), "p_reinit_60s_between_groups has no finite values"

    nonaf_path = _require_path(run_dir / "secondary" / "shdb_non_af_with_af.csv", "secondary/shdb_non_af_with_af.csv")
    nonaf = pd.read_csv(nonaf_path)
    assert len(nonaf) == 33, f"Expected 33 rows in shdb_non_af_with_af.csv, got {len(nonaf)}"
    unann_nonaf = nonaf[nonaf["Annotated"].astype(str).str.lower() == "false"]
    assert len(unann_nonaf) > 0 and unann_nonaf["n_g0_af_runs"].isna().all(), (
        "Unannotated SHDB non-AF records must have NA n_g0_af_runs (no zero fallback)"
    )

    circ_path = _require_path(run_dir / "secondary" / "circadian_analysis.csv", "secondary/circadian_analysis.csv")
    circ = pd.read_csv(circ_path)
    assert "cohort" in circ.columns, f"Missing cohort column in {list(circ.columns)}"
    circ_cohorts = set(circ["cohort"].astype(str).str.upper().unique())
    for c in ["IRIDIA", "SHDB", "LTAFDB"]:
        assert c in circ_cohorts, f"Missing cohort {c} in circadian_analysis.csv (found {circ_cohorts})"

    print(f"\n[Test 11] Aim 4 & Secondary fixes verified; prior run untouched; active run={run_dir.name}")


# ---------------------------------------------------------------------------
# Test 12: Five-Output Synthesis & Integrity Checks
# ---------------------------------------------------------------------------
def test_12_five_outputs_and_integrity() -> None:
    run_dir = _latest_run_dir()

    # 1. Output #1: within_between_decomposition.csv
    wb_path = _require_path(run_dir / "aim3" / "within_between_decomposition.csv", "within_between_decomposition.csv")
    wb = pd.read_csv(wb_path)
    assert len(wb) > 0, "within_between_decomposition.csv is empty"

    # 2. Output #5: pmva_specification_curve.csv
    pmv_path = _require_path(run_dir / "aim3" / "pmva_specification_curve.csv", "pmva_specification_curve.csv")
    pmv = pd.read_csv(pmv_path)
    assert len(pmv) > 0, "pmva_specification_curve.csv is empty"

    # 3. Left-truncated >=60s SR-reinitiation frailty shape in two_state_summary.csv
    ts_path = _require_path(run_dir / "aim3" / "two_state_summary.csv", "two_state_summary.csv")
    ts = pd.read_csv(ts_path)
    assert "sr_reinit_lt60_shape_k" in ts.columns, f"Missing sr_reinit_lt60_shape_k in {list(ts.columns)}"

    # 4. No fabricated Spitaler array anywhere in scripts/ or reports/
    fabricated_sig = "0.82, 0.77, 0.74, 0.71"
    fab_hits: list[str] = []
    for search_dir in [SCRIPTS_DIR, ROOT / "reports"]:
        if not search_dir.exists():
            continue
        for p in sorted(search_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() in {".png", ".pdf", ".parquet", ".pyc"}:
                continue
            try:
                if fabricated_sig in p.read_text(encoding="utf-8", errors="ignore"):
                    fab_hits.append(str(p.relative_to(ROOT)))
            except Exception:
                continue
    assert not fab_hits, f"Fabricated Spitaler array found in: {fab_hits}"

    # 5. Explicit Ethics & Data Licensing statement in ENTRENCH_AF_Report.md
    report_path = _require_path(ROOT / "reports" / "ENTRENCH_AF_Report.md", "ENTRENCH_AF_Report.md")
    report_txt = report_path.read_text(encoding="utf-8")
    assert "CC BY 4.0" in report_txt and "ODC-By" in report_txt, (
        "Missing explicit CC BY 4.0 / ODC-By Ethics & Data Licensing statement in ENTRENCH_AF_Report.md"
    )

    # 6. All immutable prior run directories intact
    prior_runs = [
        "run_20261126T193111Z_cc84d8b0",
        "run_20261126T200413Z_cc84d8b0",
        "run_20261126T201657Z_cc84d8b0",
        "run_20261127T174327Z_cc84d8b0",
        "run_20261127T194510Z_cc84d8b0",
        "run_20261127T200007Z_cc84d8b0",
        "run_20261128T001937Z_cc84d8b0",
    ]
    for pr in prior_runs:
        assert (RESULTS_DIR / pr).exists(), f"Prior run dir missing: {pr}"
        assert run_dir.name != pr, f"LATEST_RUN must be a fresh run dir, not {pr}"

    print(f"\n[Test 12] Five-output synthesis, licensing & 7+ immutable run dirs verified (active={run_dir.name})")


# ---------------------------------------------------------------------------
# Test 13: Epistemic Hygiene, Left-Truncated Frailty Recovery, V2 Targeted Corrections
# ---------------------------------------------------------------------------
def test_13_priorities_and_synthetic_frailty_recovery() -> None:
    run_dir = _latest_run_dir()

    # 1. Verify ZERO synthetic beat fallback in Stage 2 IRIDIA runs & exact v5 reconciliation
    runs_g0 = pd.read_parquet(_require_path(run_dir / "stage2" / "runs_g0.parquet", "runs_g0.parquet"))
    ir_g0 = runs_g0[runs_g0["cohort"] == "IRIDIA"]
    assert ir_g0["n_beats"].isna().all(), "IRIDIA n_beats must be NA when beat annotations are absent"
    assert (ir_g0["beat_source"] == "unavailable_iridia").all()
    rec_v5 = pd.read_csv(_require_path(run_dir / "stage2" / "reconciliation_v5_to_g0.csv", "reconciliation_v5_to_g0.csv"))
    assert (rec_v5["check_balance_diff"] == 0).all(), "v5->g0 reconciliation balance mismatch"
    assert {"harmonized_annotation", "continuity_nonsinus_gaps"}.issubset(set(rec_v5["tier"]))

    # 2. Verify planted-truth parameter recovery of left-truncated shared-gamma-frailty Weibull MLE
    mod7 = _load_module(_require_path(SCRIPTS_DIR / "07_aim1_termination_hazard.py", "07_aim1"), "aim1_mod")
    rng = np.random.default_rng(20260101)
    k_true, lam_true, theta_true, t_min = 0.72, 1200.0, 1.4, 300.0
    n_pats = 220
    shape_g = 1.0 / theta_true
    durs, evs, pids = [], [], []
    base_u = rng.gamma(shape=shape_g, scale=theta_true, size=n_pats)
    for p_i, u in enumerate(base_u):
        # Generate episodes conditioned on surviving past t_min
        H_min = (t_min / lam_true) ** k_true
        for _ in range(12):
            v = rng.exponential(1.0 / u)
            t_ev = lam_true * (H_min + v) ** (1.0 / k_true)
            c_t = t_min + rng.uniform(600.0, 20000.0)
            durs.append(min(t_ev, c_t))
            evs.append(1.0 if t_ev <= c_t else 0.0)
            pids.append(f"pt_{p_i:03d}")
    rec_fit = mod7.fit_shared_gamma_frailty_weibull(np.array(durs), np.array(evs), np.array(pids), t_min=t_min)
    assert rec_fit["converged"], "Left-truncated shared-gamma-frailty Weibull failed on planted synthetic data"
    assert abs(rec_fit["shape_k"] - k_true) < 0.12, f"Planted k={k_true}, recovered {rec_fit['shape_k']}"
    assert abs(rec_fit["frailty_theta"] - theta_true) < 0.45, f"Planted theta={theta_true}, recovered {rec_fit['frailty_theta']}"

    # 3. Priority 3 & V2 Correction 4: Onset audit, matched landmark denominator invariant, & dual patient medians
    oa = pd.read_csv(_require_path(run_dir / "aim1" / "onset_censoring_audit.csv", "onset_censoring_audit.csv"))
    pool_oa = oa[oa["cohort"] == "POOLED"].iloc[0]
    assert pool_oa["n_runs_g0_total"] == pool_oa["n_excluded_unknown_onset"] + pool_oa["n_analyzed_known_onset"]
    assert pool_oa["n_excluded_unknown_onset"] > 0
    mc = pd.read_csv(_require_path(run_dir / "aim1" / "model_comparison_frailty.csv", "model_comparison_frailty.csv"))
    assert "IRIDIA_primary_PAF" in set(mc["stratum"]) and mc["converged"].all()
    assert "LTAFDB_harmonized" in set(mc["stratum"]) and "LTAFDB_continuity_nonsinus_gaps" in set(mc["stratum"])
    sh = pd.read_csv(_require_path(run_dir / "aim1" / "table1_short_horizons.csv", "table1_short_horizons.csv"))
    assert "n_patients_pw" in sh.columns
    assert (sh["n_patients_pw"] <= sh["n_patients_at_risk"]).all(), "Landmark patient-weighted count exceeds risk set"
    hier = pd.read_csv(_require_path(run_dir / "aim1" / "cohort_hierarchy_summary.csv", "cohort_hierarchy_summary.csv"))
    assert "patient_weighted_quantile_median_sec" in hier.columns and "patient_median_of_medians_sec" in hier.columns

    # 4. V2 Corrections 1-3: Clipped L_W <= A_W invariant, matched eligibility ledger, & paired aggregation bootstrap
    a2r = pd.read_csv(_require_path(run_dir / "aim2" / "native_to_recon_reconciliation.csv", "native_to_recon_reconciliation.csv"))
    assert (a2r["balance_diff"] == 0).all()
    me = pd.read_csv(_require_path(run_dir / "aim2" / "matched_eligibility_reconciliation.csv", "matched_eligibility_reconciliation.csv"))
    assert (me["balance_diff"] == 0).all()
    pb = pd.read_csv(_require_path(run_dir / "aim2" / "paired_aggregation_bootstrap.csv", "paired_aggregation_bootstrap.csv"))
    assert len(pb) > 0 and (pb["n_boot_converged_both"] > 0).all()
    td = pd.read_csv(_require_path(run_dir / "aim2" / "threshold_disagreement.csv", "threshold_disagreement.csv"))
    assert {30, 360, 3600, 21600, 86400}.issubset(set(td["threshold_sec"].astype(int)))
    assert (td["n_windows_native_only_ge_T"] == 0).all(), "Clipped L_W > A_W violation in threshold_disagreement"
    assert (td["n_patients_native_only_ge_T"] == 0).all(), "Patient-level L_W > A_W violation"

    # 5. V2 Correction 5: ICC bootstrap disclosure of undefined zero-variance resamples
    icc = pd.read_csv(_require_path(run_dir / "secondary" / "iridia_repeat_icc.csv", "iridia_repeat_icc.csv"))
    icc_r = icc[icc["metric"] == "reinit_360s_prop"].iloc[0]
    assert icc_r["n_boot_undefined_zero_var"] > 0.05 * icc_r["n_boot_total"]
    assert pd.isna(icc_r["icc_boot_ci_lo"]) and pd.isna(icc_r["icc_boot_ci_hi"])

    # 6. Priority 7 & V2 Correction 6: Principal HRS figure, updated title, & Blinded EP Review Packet
    hrs_fig = _require_path(ROOT / "reports" / "figures" / "HRS_Principal_Figure.png", "HRS_Principal_Figure.png")
    assert hrs_fig.stat().st_size > 10000
    hrs_abs = _require_path(ROOT / "reports" / "HRS_2027_Abstract.md", "HRS_2027_Abstract.md").read_text(encoding="utf-8")
    assert "Temporal Aggregation Alters Inferred Atrial Fibrillation Termination Dynamics" in hrs_abs
    ep_pkt = _require_path(ROOT / "reports" / "EP_Review_Packet_Cheung.md", "EP_Review_Packet_Cheung.md")
    assert "INDETERMINATE" in ep_pkt.read_text(encoding="utf-8")
    ep_sheet = pd.read_csv(_require_path(ROOT / "reports" / "ep_review_sheet.csv", "ep_review_sheet.csv"))
    assert len(ep_sheet) >= 18
    assert "cohort" not in ep_sheet.columns and "next_rhythm_native" not in ep_sheet.columns, "Blinded sheet leaked ground truth"
    ep_key = pd.read_csv(_require_path(ROOT / "reports" / "ep_review_key_SEALED.csv", "ep_review_key_SEALED.csv"))
    assert len(ep_key) == len(ep_sheet)

    print("\n[Test 13] Zero synthetic fallbacks, left-truncated frailty recovery, & V2 HRS targeted corrections verified")


# ---------------------------------------------------------------------------
# Test 14: Final Reproducibility Freeze & V2 Audit Requirements (9.1-9.6)
# ---------------------------------------------------------------------------
def test_14_reproducibility_freeze_and_v2_audit() -> None:
    run_dir = _latest_run_dir()

    # 1. Exact 2195 == 2195 reconciliation (zero 31-run cross-cohort record collision)
    rec_aud = pd.read_csv(_require_path(run_dir / "aim2" / "native_to_recon_reconciliation.csv", "native_to_recon_reconciliation.csv"))
    aud_pool = rec_aud[(rec_aud["mode"] == "recording_24h_windows") & (rec_aud["stratum"] == "POOLED")].iloc[0]
    n_aud_elig = int(aud_pool["retained_single_run"]) + int(aud_pool["merged_multi_run"])
    me = pd.read_csv(_require_path(run_dir / "aim2" / "matched_eligibility_reconciliation.csv", "matched_eligibility_reconciliation.csv"))
    me_pool = me[me["cohort"] == "POOLED"].iloc[0]
    assert int(me_pool["n_runs_in_eligible_windows"]) == n_aud_elig, (
        f"Matched eligibility runs ({me_pool['n_runs_in_eligible_windows']}) != ledger retained+merged ({n_aud_elig})"
    )
    ledger = pd.read_csv(_require_path(run_dir / "aim2" / "matched_eligibility_run_ledger.csv", "matched_eligibility_run_ledger.csv"))
    assert len(ledger) == n_aud_elig and ledger["run_id"].nunique() == n_aud_elig

    # 2. Coverage sensitivity (>=95% and >=99% known rhythm coverage)
    ca = pd.read_csv(_require_path(run_dir / "aim2" / "coverage_sensitivity_aggregation.csv", "coverage_sensitivity_aggregation.csv"))
    assert {"all", "cov_ge_95pct", "cov_ge_99pct"}.issubset(set(ca["coverage_filter"]))
    ctd = pd.read_csv(_require_path(run_dir / "aim2" / "coverage_sensitivity_threshold.csv", "coverage_sensitivity_threshold.csv"))
    assert {"all", "known_cov_ge_95pct", "known_cov_ge_99pct"}.issubset(set(ctd["coverage_filter"]))
    assert (ctd["n_windows_native_only_ge_T"] == 0).all()

    # 3. Common >=6-min landmark & delayed-entry likelihood synthetic verification
    pb = pd.read_csv(_require_path(run_dir / "aim2" / "paired_aggregation_bootstrap.csv", "paired_aggregation_bootstrap.csv"))
    m360 = pb[pb["min_dur_sec"] >= 360.0]
    assert len(m360) > 0 and (m360["cond_landmark_sec"] == 360.0).all()
    assert m360["delta_unfrailty_k_ci_lo"].notna().all()

    de = pd.read_csv(_require_path(run_dir / "aim1" / "delayed_entry_likelihood_verification.csv", "delayed_entry_likelihood_verification.csv"))
    assert {"A_conditional_delayed_entry_lt300s", "B_single_pop_survival_selection_30s_24h", "C_heterogeneous_mixture_selection_30s_24h"}.issubset(set(de["fixture"]))
    assert de["converged"].all()
    row_a = de[de["fixture"] == "A_conditional_delayed_entry_lt300s"].iloc[0]
    assert abs(float(row_a["bias_k_lt"])) < 0.08 and abs(float(row_a["bias_k_naive"])) > 0.25

    # 4. Bootstrap replicates, boundary-hit / identifiability audit, & distinct cluster check
    brep = pd.read_csv(_require_path(run_dir / "aim2" / "paired_bootstrap_replicates.csv", "paired_bootstrap_replicates.csv"))
    assert len(brep) >= 1000
    b_id = pd.read_csv(_require_path(run_dir / "aim2" / "bootstrap_identifiability_audit.csv", "bootstrap_identifiability_audit.csv"))
    lt_row = b_id[(b_id["rhythm_tier"] == "harmonized_annotation") & (b_id["cohort"] == "LTAFDB")].iloc[0]
    assert lt_row["mean_eps_per_patient_recon"] <= 1.05 and not bool(lt_row["frailty_identifiable"])

    # 5. Reproducibility freeze archive verification (if exported)
    freeze_dir = RESULTS_DIR / f"reproducibility_freeze_{run_dir.name}"
    if freeze_dir.exists():
        for req in [
            freeze_dir / "working_tree.diff",
            freeze_dir / "environment_pip_freeze.txt",
            freeze_dir / "seeds_and_hashes.json",
            freeze_dir / "input_dataset_checksums.sha256",
            freeze_dir / "deliverables" / "HRS_2027_Abstract.md",
            freeze_dir / "deliverables" / "HRS_Principal_Figure.png",
            freeze_dir / "deliverables" / "artifacts" / "paired_bootstrap_replicates.csv",
        ]:
            assert req.exists(), f"Freeze artifact missing: {req}"

    print("\n[Test 14] Final reproducibility freeze & V2 audit requirements (9.1-9.6) verified")
