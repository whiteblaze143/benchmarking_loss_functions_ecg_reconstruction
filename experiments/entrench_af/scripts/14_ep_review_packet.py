#!/usr/bin/env python3
"""
Stage 14 — Blinded Electrophysiologist ECG Review Packet (Priority 7).

Builds a blinded multi-cohort rhythm-transition adjudication packet for
Dr. Christopher Cheung:
  - 24 transitions stratified across IRIDIA NON_AF (binary label),
    LTAFDB (N, SBR, and non-sinus VT/B/IVR/SVTA), and SHDB-AF (N vs AFL/AT).
  - High-resolution ECG strips [-12 s, +18 s] around AF termination (t = 0)
    with standard ECG paper grid and ZERO dataset or ground-truth labels.
  - Explicit `INDETERMINATE` response option on both Markdown form and CSV sheet.
  - Sealed unblinding key saved separately (`reports/ep_review_key_SEALED.csv`).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

SEED = 20260101
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import common  # type: ignore

spec5 = importlib.util.spec_from_file_location("stage5_mod", SCRIPTS_DIR / "05_markers_terminal_af.py")
S5 = importlib.util.module_from_spec(spec5)
assert spec5.loader is not None
spec5.loader.exec_module(S5)


def _bandpass_display(ecg: np.ndarray, fs: float) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float64)
    ecg = np.nan_to_num(ecg, nan=0.0)
    ecg = ecg - np.median(ecg)
    b, a = signal.butter(2, [0.5 / (0.5 * fs), min(35.0, 0.45 * fs) / (0.5 * fs)], btype="band")
    return signal.filtfilt(b, a, ecg)


def _render_blinded_strip(
    blinded_id: str,
    ecg: np.ndarray,
    fs: float,
    pre_sec: float,
    post_sec: float,
    out_path: Path,
) -> None:
    t = np.arange(len(ecg), dtype=float) / fs - pre_sec
    sig = _bandpass_display(ecg, fs)
    scale = float(np.percentile(np.abs(sig), 99.0))
    if scale > 0:
        sig = np.clip(sig, -1.35 * scale, 1.35 * scale)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 4.6), gridspec_kw={"height_ratios": [1, 1]})
    for ax, (t0, t1), row_lbl in [
        (ax1, (-pre_sec, min(post_sec, 3.0)), f"{blinded_id} — Pre-Transition & Annotated Boundary (t = 0 s)"),
        (ax2, (max(-2.0, -pre_sec), post_sec), f"{blinded_id} — Post-Transition Rhythm Continuation"),
    ]:
        m = (t >= t0) & (t <= t1)
        ax.plot(t[m], sig[m], color="#111111", linewidth=0.95)
        ax.axvline(0.0, color="#B22222", linestyle="--", linewidth=1.3, label="Annotated termination (t = 0 s)")
        ax.set_xlim(t0, t1)
        ax.set_xticks(np.arange(np.floor(t0), np.ceil(t1) + 0.1, 1.0))
        ax.grid(True, which="major", color="#E5A8A8", linewidth=0.55, alpha=0.65)
        ax.set_title(row_lbl, fontsize=9.5, loc="left", pad=4)
        ax.set_ylabel("Amplitude (filtered)", fontsize=8.5)
        ax.tick_params(labelsize=8)
    ax2.set_xlabel("Time relative to annotated AF offset (s)", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_dir = common.resolve_run_dir()
    runs = pd.read_parquet(run_dir / "stage2" / "runs_g0.parquet")
    events = pd.read_parquet(run_dir / "stage4" / "events_g0.parquet")
    tm = pd.read_parquet(run_dir / "stage5" / "terminal_markers.parquet")[["event_id", "sqi_pass_terminal"]]
    pm = pd.read_parquet(run_dir / "stage6" / "post_markers.parquet")[["event_id", "sqi_pass_postA"]]

    df = events.merge(tm, on="event_id", how="left").merge(pm, on="event_id", how="left")
    # Also attach any non-censored runs with next_rhythm_class from runs_g0 if needed (events already has dur>=30s)
    df_clean = df[(df["sqi_pass_terminal"] == 1) & (df["sqi_pass_postA"] == 1) & (df["followup_rem_sec"] >= 20.0)].copy()

    rng = np.random.default_rng(SEED)
    pre_sec, post_sec = 12.0, 18.0

    # Target stratified quota (24 cases):
    # - 8 IRIDIA (4 rapid reinit <360s, 4 non-early >=1800s)
    # - 8 LTAFDB (3 N sinus controls, 2 SBR sinus bradycardia, 3 non-sinus VT/B/IVR/T/SVTA)
    # - 8 SHDB   (4 N sinus controls, 4 AFL/AT/AB organized/ectopic interruptions)
    ir = df_clean[df_clean["cohort"] == "IRIDIA"]
    lt = df_clean[df_clean["cohort"] == "LTAFDB"]
    sh = df_clean[df_clean["cohort"] == "SHDB"]

    # Prefer unique patients per stratum where possible
    def _pick(sub: pd.DataFrame, k: int, tag: str) -> pd.DataFrame:
        if sub.empty:
            return sub
        sub_dedup = sub.sample(frac=1.0, random_state=SEED).drop_duplicates("patient")
        if len(sub_dedup) >= k:
            out = sub_dedup.head(k).copy()
        else:
            rem = sub[~sub.index.isin(sub_dedup.index)].head(max(0, k - len(sub_dedup)))
            out = pd.concat([sub_dedup, rem], ignore_index=False).head(k).copy()
        out["sampling_stratum"] = tag
        return out

    picks = [
        _pick(ir[(ir["event"] == 1) & (ir["time_to_reinit_sec"] < 360.0)], 4, "IRIDIA_NONAF_rapid_reinit"),
        _pick(ir[ir["time_to_reinit_sec"] >= 1800.0], 4, "IRIDIA_NONAF_sustained_pause"),
        _pick(lt[lt["next_rhythm_native"] == "N"], 3, "LTAFDB_sinus_N"),
        _pick(lt[lt["next_rhythm_native"] == "SBR"], 2, "LTAFDB_sinus_brady_SBR"),
        _pick(lt[lt["next_rhythm_class"].isin(["VENTRICULAR_OR_ECTOPIC", "ORGANIZED_ATRIAL"])], 3, "LTAFDB_nonsinus_interruption"),
        _pick(sh[sh["next_rhythm_native"] == "N"], 4, "SHDB_sinus_N"),
        _pick(sh[sh["next_rhythm_class"].isin(["ORGANIZED_ATRIAL", "VENTRICULAR_OR_ECTOPIC"])], 4, "SHDB_atrial_or_ectopic_interruption"),
    ]
    pool = pd.concat([p for p in picks if not p.empty], ignore_index=True)

    # Shuffle order across all strata so case numbers leak zero cohort/rhythm ordering
    pool = pool.sample(frac=1.0, random_state=SEED + 7).reset_index(drop=True)

    fig_dir = PROJECT_ROOT / "reports" / "figures" / "ep_review"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old_png in fig_dir.glob("EP_*.png"):
        old_png.unlink()

    blinded_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    readers: dict[tuple[str, str], Any] = {}

    case_idx = 0
    for r in pool.itertuples(index=False):
        key = (str(r.cohort), str(r.record))
        if key not in readers:
            readers[key] = S5.RecordReader(str(r.cohort), str(r.record))
        reader = readers[key]
        t0 = float(r.t0)
        if t0 < pre_sec:
            continue
        ecg = reader.read_segment(t0 - pre_sec, t0 + post_sec)
        if ecg is None or len(ecg) < int(reader.fs * (pre_sec + post_sec - 1.0)):
            continue

        case_idx += 1
        bid = f"EP_{case_idx:02d}"
        img_rel = f"figures/ep_review/{bid}.png"
        _render_blinded_strip(bid, ecg, reader.fs, pre_sec, post_sec, PROJECT_ROOT / "reports" / img_rel)

        blinded_rows.append({
            "blinded_id": bid,
            "strip_image_relpath": img_rel,
            "pre_transition_sec": pre_sec,
            "post_transition_sec": post_sec,
            "adjudicated_post_rhythm": "",
            "genuine_af_termination": "",
            "reviewer_confidence_1_to_3": "",
            "notes": "",
        })
        sealed_rows.append({
            "blinded_id": bid,
            "sampling_stratum": r.sampling_stratum,
            "cohort": r.cohort,
            "record": r.record,
            "patient": r.patient,
            "event_id": r.event_id,
            "t0_sec": t0,
            "af_dur_sec": float(r.dur_sec),
            "next_rhythm_native": r.next_rhythm_native,
            "next_rhythm_class": r.next_rhythm_class,
            "time_to_reinit_sec": float(r.time_to_reinit_sec),
            "reinit_observed": int(r.event),
        })

    sheet_df = pd.DataFrame(blinded_rows)
    key_df = pd.DataFrame(sealed_rows)

    sheet_path = PROJECT_ROOT / "reports" / "ep_review_sheet.csv"
    key_path = PROJECT_ROOT / "reports" / "ep_review_key_SEALED.csv"
    sheet_df.to_csv(sheet_path, index=False)
    key_df.to_csv(key_path, index=False)

    md_lines = [
        "# Blinded Electrophysiologist ECG Adjudication Packet",
        "",
        "**Prepared for:** Dr. Christopher Cheung (Cardiac Electrophysiology Review)",
        f"**Cases Included:** {len(sheet_df)} blinded rhythm-transition strips (`[-12 s, +18 s]` around annotated AF termination `t = 0 s`, single surface ECG lead, bandpass filtered 0.5–35 Hz)",
        "**Blinding:** Cohort source, patient identifier, upstream rhythm label, and downstream reinitiation timing are completely withheld from all strip images and the adjudication sheet (`ep_review_sheet.csv`).",
        "",
        "## Adjudication Instructions",
        "For each strip (`EP_01` through `EP_{:02d}`), classify:".format(len(sheet_df)),
        "1. **`adjudicated_post_rhythm`** (select one):",
        "   - `CONFIRMED_SINUS`: Clear sinus P waves / regular sinus rhythm restored after `t = 0 s`",
        "   - `NON_SINUS_ATRIAL_AFL_AT`: Organized atrial arrhythmia (atrial flutter or atrial tachycardia) without sinus restoration",
        "   - `VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION`: Ventricular rhythm, bigeminy/trigeminy, or ectopy interruption while underlying atrial fibrillation likely persists",
        "   - `ARTIFACT_OR_NOISE`: Signal dropout, motion artifact, or baseline shift mimicking rhythm transition",
        "   - `INDETERMINATE`: Cannot reliably determine post-transition rhythm from available single-lead strip",
        "2. **`genuine_af_termination`** (`YES`, `NO`, or `INDETERMINATE`)",
        "3. **`reviewer_confidence_1_to_3`** (`1` = low, `2` = moderate, `3` = high)",
        "",
        "---",
        "",
        "## Blinded ECG Strips",
        "",
    ]
    for r in blinded_rows:
        md_lines.extend([
            f"### {r['blinded_id']}",
            f"![{r['blinded_id']}]({r['strip_image_relpath']})",
            "- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`",
            "- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`",
            "",
        ])

    packet_path = PROJECT_ROOT / "reports" / "EP_Review_Packet_Cheung.md"
    packet_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[Stage 14] Built blinded EP review packet: {len(sheet_df)} strips -> {packet_path}, {sheet_path}, and sealed key {key_path}")


if __name__ == "__main__":
    main()
