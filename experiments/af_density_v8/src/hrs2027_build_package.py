"""Build the Heart Rhythm 2027 Clinical Adjudication Package & Composite Figure 1.

Strict constraints enforced:
- Uses frozen V8 outputs only (no new sweeps/models/exploratory fishing).
- Authentic original LTAFDB WFDB waveforms (both channels, 128.0 Hz, physical mV,
  raw unfiltered signal with only DC window-median baseline offset removed).
- Reviewer-facing PDFs and Excel review form strictly blinded (randomized Case_IDs
  and presentation order; zero native WFDB tags, automated labels, record IDs,
  density metrics, or impact ranks revealed).
- Diagnostic panels calibrated physically to 25.0 mm/s (plus explicit 10/20/5 mm/mV
  gain with 1.0 mV x 0.20 s calibration pulse on a standard 1 mm / 5 mm ECG grid).
"""
import hashlib
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image
import pandas as pd
import wfdb

sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import clip_ivs_to_known
from v8_exp_a_reference_layers import metrics_from_intervals

LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
RES_DIR = Path("experiments/af_density_v8/results")
OUT_ROOT = Path("reports/HRS2027")
EP_DIR = OUT_ROOT / "HRS2027_EP_Review"
INV_DIR = OUT_ROOT / "investigator_only"

FREEZE_TS_UTC = "2026-11-25T05:30:00Z"
SEED = 20270501


def resolve_rec_path(rid: str) -> str:
    s = str(int(rid)) if str(rid).isdigit() else str(rid)
    if (LTAF_DIR / f"{s}.hea").exists():
        return s
    s2 = s.zfill(2)
    if (LTAF_DIR / f"{s2}.hea").exists():
        return s2
    raise FileNotFoundError(f"Missing LTAFDB record {rid}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_frozen_selection():
    """Freeze Batch A (19), Batch B (39), and disjoint Reference Controls (8)."""
    ep = pd.read_csv(RES_DIR / "V8_rev3_ep_review_packet_top60.csv")
    assert len(ep) == 58 and ep["record_id"].nunique() == 19

    # Prespecified Batch A rule: highest single_gap_abs_delta_D per patient
    # (fallback pct_of_total_r0_nonaf when single-gap ASSUMED_ALL_AF has NaN delta_D)
    ep_sorted = ep.sort_values(
        ["record_id", "single_gap_abs_delta_D", "pct_of_total_r0_nonaf", "dur_s", "gap_start_s"],
        ascending=[True, False, False, False, True],
        na_position="last",
    ).copy()
    first_idx = ep_sorted.groupby("record_id", sort=False).head(1).index
    ep_sorted["is_batch_a"] = ep_sorted.index.isin(first_idx)

    ba = ep_sorted[ep_sorted["is_batch_a"]].copy().reset_index(drop=True)
    bb = ep_sorted[~ep_sorted["is_batch_a"]].copy().reset_index(drop=True)
    assert len(ba) == 19 and len(bb) == 39

    # Verify disjoint reference control segments (4 clear AF, 4 clear organized sinus)
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    lr = rhy[(rhy["dataset"] == "LTAFDB") & (rhy["rhythm_known"])].copy()
    hb19_set = set(ep["record_id"].astype(str))
    ctrl_specs = [
        ("70", 18000.0, 18005.0, "CONTROL_CLEAR_AF", "AFIB"),
        ("18", 18000.0, 18005.0, "CONTROL_CLEAR_AF", "AFIB"),
        ("54", 18000.0, 18005.0, "CONTROL_CLEAR_AF", "AFIB"),
        ("17", 18000.0, 18005.0, "CONTROL_CLEAR_AF", "AFIB"),
        ("117", 40000.0, 40005.0, "CONTROL_CLEAR_SINUS", "N"),
        ("42", 35000.0, 35005.0, "CONTROL_CLEAR_SINUS", "N"),
        ("06", 35000.0, 35005.0, "CONTROL_CLEAR_SINUS", "N"),
        ("08", 40000.0, 40005.0, "CONTROL_CLEAR_SINUS", "N"),
    ]
    ctrl_rows = []
    for rid, gs, ge, ctype, exp_native in ctrl_specs:
        assert rid not in hb19_set and rid.lstrip("0") not in hb19_set
        sub = lr[lr["record_id"].astype(str) == rid]
        match = sub[(sub["start_time_s"] <= gs - 600.0) & (sub["end_time_s"] >= ge + 600.0)]
        assert len(match) == 1 and match.iloc[0]["rhythm_label_native"] == exp_native, f"Control verify failed {rid}"
        ctrl_rows.append({
            "record_id": rid,
            "gap_start_s": gs,
            "gap_end_s": ge,
            "dur_s": ge - gs,
            "native_tags": exp_native,
            "is_native_ventricular": False,
            "r1_class": ctype,
            "r0_density": np.nan,
            "n_beats": np.nan,
            "n_N": np.nan,
            "n_V": np.nan,
            "cvrr_gap": np.nan,
            "f_power_ratio": np.nan,
            "p_sync_snr": np.nan,
            "single_gap_bridge_state": ctype,
            "single_gap_abs_delta_D": np.nan,
            "single_gap_abs_delta_G": np.nan,
            "pct_of_total_r0_nonaf": np.nan,
            "control_truth": ctype,
        })
    bc = pd.DataFrame(ctrl_rows)

    rng = np.random.default_rng(SEED)
    all_rows = []
    for batch_code, prefix, b_label, df_b in [
        ("BATCH_A", "CASE-A", "Batch A (Primary 19)", ba),
        ("BATCH_B", "CASE-B", "Batch B (Additional 39)", bb),
        ("CONTROL", "CTRL-", "Reference Controls (8)", bc),
    ]:
        perm = rng.permutation(len(df_b))
        df_shuf = df_b.iloc[perm].reset_index(drop=True)
        for pos, row in df_shuf.iterrows():
            cid = f"{prefix}{pos + 1:02d}"
            rid_str = resolve_rec_path(str(row["record_id"]))
            d = row.to_dict()
            d.update({
                "case_id": cid,
                "batch_code": batch_code,
                "batch_label": b_label,
                "presentation_order": pos + 1,
                "wfdb_record_id": rid_str,
                "patient_id": f"ltafdb_{rid_str}",
                "control_truth": d.get("control_truth", "TARGET_INTERRUPTION"),
                "selection_freeze_utc": FREEZE_TS_UTC,
            })
            all_rows.append(d)

    full = pd.DataFrame(all_rows)
    assert len(full) == 66
    return full


def extract_case_waveform(row):
    """Load authentic raw WFDB physical mV signal (~30s pre, target, ~30s post) + SHA256."""
    rid = row["wfdb_record_id"]
    gs, ge = float(row["gap_start_s"]), float(row["gap_end_s"])
    rec_path = LTAF_DIR / rid
    hdr = wfdb.rdheader(str(rec_path))
    fs = float(hdr.fs)
    pre_s, post_s = 30.0, 30.0
    sf = max(0, int(round((gs - pre_s) * fs)))
    st = min(int(hdr.sig_len), int(round((ge + post_s) * fs)))
    rec = wfdb.rdrecord(str(rec_path), sampfrom=sf, sampto=st)
    raw_sig = rec.p_signal.astype(np.float64)
    assert np.isfinite(raw_sig).all(), f"NaN in {rid}"

    # Median DC offset per channel only; preserves exact raw AC amplitude and f/P waves.
    sig_centered = raw_sig - np.median(raw_sig, axis=0, keepdims=True)
    t_rel = (np.arange(sf, st) / fs) - gs

    sig_hash = hashlib.sha256(np.ascontiguousarray(raw_sig).tobytes()).hexdigest()
    hash_info = {
        "case_id": row["case_id"],
        "wfdb_record_id": rid,
        "fs_hz": fs,
        "sig_name": "|".join(rec.sig_name),
        "units": "|".join(rec.units),
        "adc_gain": "|".join(str(x) for x in rec.adc_gain),
        "baseline": "|".join(str(x) for x in rec.baseline),
        "sampfrom": sf,
        "sampto": st,
        "n_samples": int(len(raw_sig)),
        "p_signal_sha256": sig_hash,
        "hea_sha256": sha256_file(LTAF_DIR / f"{rid}.hea"),
        "dat_sha256": sha256_file(LTAF_DIR / f"{rid}.dat"),
        "atr_sha256": sha256_file(LTAF_DIR / f"{rid}.atr"),
    }
    return t_rel, sig_centered, fs, rec.sig_name, hash_info


def choose_gain_mm_per_mv(sig_ch: np.ndarray) -> tuple:
    """Standard 10 mm/mV unless trace overflows +/-1.45 mV (5 mm/mV) or is tiny <0.45 mV (20 mm/mV)."""
    max_abs = float(np.max(np.abs(sig_ch)))
    ptp_core = float(np.percentile(sig_ch, 99.5) - np.percentile(sig_ch, 0.5))
    if max_abs > 1.48:
        return 5.0, (-3.0, 3.0), 0.20, 1.00
    if ptp_core < 0.45 and max_abs < 0.72:
        return 20.0, (-0.75, 0.75), 0.05, 0.25
    return 10.0, (-1.5, 1.5), 0.10, 0.50


def draw_ecg_grid_panel(ax, t_rel, sig_1ch, t_win_start, t_win_end, dur_s,
                        ch_label, gain_mm_mv, ylim, minor_mv, major_mv,
                        show_cal_pulse=True, draw_xlabel=False):
    """Render a calibrated diagnostic strip (10.0 s trace + 0.50 s cal margin = 25 mm/s physical width)."""
    # Reserve 0.50 s left margin exclusively for the 1.0 mV x 0.20 s calibration pulse
    cal_pad = 0.50
    x0 = t_win_start - cal_pad
    x1 = t_win_end
    ax.set_xlim(x0, x1)
    ax.set_ylim(ylim[0], ylim[1])

    # Clinical pink ECG grid (1 mm minor = 0.04 s x minor_mv; 5 mm major = 0.20 s x major_mv)
    ax.set_facecolor("#FFF9FA")
    minor_x = np.arange(np.floor(x0 / 0.04) * 0.04, x1 + 0.001, 0.04)
    major_x = np.arange(np.floor(x0 / 0.20) * 0.20, x1 + 0.001, 0.20)
    minor_y = np.arange(ylim[0], ylim[1] + 1e-9, minor_mv)
    major_y = np.arange(ylim[0], ylim[1] + 1e-9, major_mv)
    for xv in minor_x:
        ax.axvline(xv, color="#FAD4D8", lw=0.35, zorder=0)
    for yv in minor_y:
        ax.axhline(yv, color="#FAD4D8", lw=0.35, zorder=0)
    for xv in major_x:
        ax.axvline(xv, color="#F09AA2", lw=0.65, zorder=0.5)
    for yv in major_y:
        ax.axhline(yv, color="#F09AA2", lw=0.65, zorder=0.5)

    # Target interval highlight & unambiguous onset/offset markers
    ov_a = max(t_win_start, 0.0)
    ov_b = min(t_win_end, dur_s)
    if ov_b > ov_a:
        ax.axvspan(ov_a, ov_b, color="#FDE68A", alpha=0.22, zorder=1)
    if t_win_start <= 0.0 <= t_win_end:
        ax.axvline(0.0, color="#B91C1C", ls="--", lw=1.25, zorder=3)
        onset_ha = "left" if (0.0 - t_win_start) < 1.15 else "right"
        onset_x = 0.05 if onset_ha == "left" else -0.05
        ax.text(onset_x, ylim[1] * 0.87, "ONSET (0.00 s)", color="#991B1B",
                fontsize=7.0, fontweight="bold", ha=onset_ha, va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#FCA5A5", alpha=0.92))
    if t_win_start <= dur_s <= t_win_end:
        ax.axvline(dur_s, color="#1D4ED8", ls="--", lw=1.25, zorder=3)
        off_ha = "right" if (t_win_end - dur_s) < 1.25 else "left"
        off_x = dur_s - 0.05 if off_ha == "right" else dur_s + 0.05
        ax.text(off_x, ylim[1] * 0.87, f"OFFSET (+{dur_s:.2f} s)", color="#1E3A8A",
                fontsize=7.0, fontweight="bold", ha=off_ha, va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#93C5FD", alpha=0.92))

    # Draw 1.0 mV x 0.20 s standard rectangular calibration pulse in left margin
    if show_cal_pulse:
        cp_base = -0.5 * (ylim[1] - ylim[0]) * 0.35
        if cp_base + 1.0 > ylim[1] * 0.90:
            cp_base = ylim[0] * 0.70
        cal_x = [x0 + 0.05, x0 + 0.14, x0 + 0.14, x0 + 0.34, x0 + 0.34, x0 + 0.43]
        cal_h = 1.0 if (cp_base + 1.0 <= ylim[1] * 0.92) else 0.5
        cal_y = [cp_base, cp_base, cp_base + cal_h, cp_base + cal_h, cp_base, cp_base]
        ax.plot(cal_x, cal_y, color="#111827", lw=1.35, zorder=4)
        ax.text(x0 + 0.24, cp_base - (ylim[1] - ylim[0]) * 0.08, f"{cal_h:.1f} mV\n0.20 s",
                fontsize=5.8, ha="center", va="top", color="#111827", zorder=5)

    # Actual raw ECG waveform inside [t_win_start, t_win_end]
    m = (t_rel >= t_win_start) & (t_rel <= t_win_end)
    ax.plot(t_rel[m], sig_1ch[m], color="#0F172A", lw=0.88, zorder=4)

    gain_tag = f"{int(gain_mm_mv)} mm/mV" + (" (2x gain)" if gain_mm_mv == 20.0 else (" (0.5x gain)" if gain_mm_mv == 5.0 else ""))
    ax.text(t_win_start + 0.08, ylim[0] + (ylim[1] - ylim[0]) * 0.07,
            f"{ch_label} | {gain_tag}", fontsize=7.5, fontweight="bold", color="#1E293B",
            va="bottom", zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CBD5E1", alpha=0.88))

    ax.set_yticks([])
    ticks_x = np.arange(np.ceil(t_win_start), np.floor(t_win_end) + 0.01, 1.0)
    ax.set_xticks(ticks_x)
    if draw_xlabel:
        ax.set_xticklabels([f"{x:.0f}s" for x in ticks_x], fontsize=7.5)
        ax.set_xlabel("Relative Time (s; 0.00 s = Target Onset)  —  Grid: 25.0 mm/s (1 mm = 0.04 s, 5 mm = 0.20 s)", fontsize=8.0, labelpad=2)
    else:
        ax.set_xticklabels([])
    for sp in ax.spines.values():
        sp.set_color("#94A3B8")
        sp.set_linewidth(0.8)


def render_overview_panel(ax, t_rel, sig, dur_s, title_lbl):
    """Render full ~65s+ context strip (Ch1 + Ch2) before/during/after target."""
    s0 = sig[:, 0]
    s1 = sig[:, 1]
    span0 = max(1.0, float(np.percentile(s0, 99.5) - np.percentile(s0, 0.5)))
    span1 = max(1.0, float(np.percentile(s1, 99.5) - np.percentile(s1, 0.5)))
    off0 = 0.65 * span1 + 0.55 * span0
    y0 = s0 + off0
    y1 = s1
    ymin = float(np.percentile(y1, 0.2)) - 0.35 * span1
    ymax = float(np.percentile(y0, 99.8)) + 0.35 * span0

    ax.set_xlim(t_rel[0], t_rel[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_facecolor("#F8FAFC")
    ax.axvspan(0.0, dur_s, color="#FDE68A", alpha=0.35, zorder=1)
    ax.axvline(0.0, color="#B91C1C", ls="--", lw=1.1, zorder=3)
    ax.axvline(dur_s, color="#1D4ED8", ls="--", lw=1.1, zorder=3)

    ax.plot(t_rel, y0, color="#0F172A", lw=0.62, zorder=4)
    ax.plot(t_rel, y1, color="#334155", lw=0.62, zorder=4)

    # 1.0 mV vertical reference bar cleanly on far right
    bar_x = t_rel[-1] - (t_rel[-1] - t_rel[0]) * 0.025
    bar_y0 = ymin + (ymax - ymin) * 0.08
    ax.plot([bar_x, bar_x], [bar_y0, bar_y0 + 1.0], color="#111827", lw=2.0, zorder=6)
    ax.text(bar_x - (t_rel[-1] - t_rel[0]) * 0.007, bar_y0 + 0.5, "1.0 mV", fontsize=6.8,
            fontweight="bold", ha="right", va="center", color="#111827",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#CBD5E1", alpha=0.92), zorder=6)

    ax.text(t_rel[0] + (t_rel[-1] - t_rel[0]) * 0.01, ymax - (ymax - ymin) * 0.06, "Ch1 (Overview)",
            fontsize=7.1, fontweight="bold", color="#0F172A", va="top",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#CBD5E1", alpha=0.9), zorder=6)
    ax.text(t_rel[0] + (t_rel[-1] - t_rel[0]) * 0.01, np.median(y1) + 0.42 * span1, "Ch2 (Overview)",
            fontsize=7.1, fontweight="bold", color="#334155", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#CBD5E1", alpha=0.9), zorder=6)
    ax.set_title(title_lbl, fontsize=8.5, fontweight="bold", loc="left", pad=3, color="#1E293B")
    ax.set_yticks([])
    step_s = 5.0 if (t_rel[-1] - t_rel[0]) <= 90.0 else 10.0
    xt = np.arange(np.ceil(t_rel[0] / step_s) * step_s, t_rel[-1] + 0.1, step_s)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{x:.0f}s" for x in xt], fontsize=7.2)
    ax.grid(axis="x", color="#CBD5E1", lw=0.5, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_color("#94A3B8")


def render_case_pages(pdf: PdfPages, row, t_rel, sig, fs, batch_idx, batch_total):
    """Create 1 (or 2 for long targets > 8.5s) strictly blinded Landscape US Letter pages."""
    cid = row["case_id"]
    dur_s = float(row["dur_s"])
    g0, ylim0, min0, maj0 = choose_gain_mm_per_mv(sig[:, 0])
    g1, ylim1, min1, maj1 = choose_gain_mm_per_mv(sig[:, 1])

    def new_landscape_fig(subtitle_note: str):
        # US Letter Landscape: 11.0 x 8.5 in = 279.4 x 215.9 mm
        fig = plt.figure(figsize=(11.0, 8.5))
        # Header banner (cleanly separated left title & right metadata)
        fig.text(0.028, 0.962, f"{cid}  |  {row['batch_label']}  [{batch_idx} / {batch_total}]",
                 fontsize=11.0, fontweight="bold", color="#0F172A")
        fig.text(0.972, 0.962,
                 f"Target: {dur_s:.2f} s  |  fs = {fs:.1f} Hz  |  Grid: 25.0 mm/s, Ch1 {int(g0)} mm/mV, Ch2 {int(g1)} mm/mV",
                 fontsize=8.2, ha="right", color="#334155", fontweight="bold")
        fig.text(0.028, 0.935, subtitle_note, fontsize=8.0, color="#475569")
        fig.text(0.028, 0.015,
                 "Blinded Review Strip — Unfiltered two-lead ambulatory ECG recording (128 Hz, mV; DC baseline centered only to preserve f-waves and P-waves). Shaded band = Target Interval [0.00 s, +{:.2f} s].".format(dur_s),
                 fontsize=7.3, color="#64748B")
        return fig

    # Exact physical dimensions inside 11.0 x 8.5 in figure:
    # 10.50 s total x-axis (0.50 s cal margin + 10.00 s trace) at 25.0 mm/s = 262.5 mm = 10.3346 in -> left=0.030, w=0.9395
    diag_w = 10.3346 / 11.0
    diag_l = (1.0 - diag_w) / 2.0
    ch_h = 1.1811 / 8.5  # exact 30.0 mm vertical height -> +/-1.5mV = 10mm/mV, +/-0.75mV = 20mm/mV, +/-3.0mV = 5mm/mV

    if dur_s <= 8.5:
        # Single clean page: Overview (top) + True 25 mm/s Diagnostic Strip centering entire target
        margin = max(0.75, (10.0 - dur_s) / 2.0)
        ws = max(t_rel[0] + 0.05, -margin)
        we = ws + 10.0
        if we > t_rel[-1]:
            we = t_rel[-1]
            ws = we - 10.0

        fig = new_landscape_fig("Single-Page Assessment: 60s+ Rhythm Context (Top) + Complete Enlarged Target at True 25.0 mm/s (Bottom)")
        ax_ov = fig.add_axes([diag_l, 0.57, diag_w, 0.30])
        render_overview_panel(ax_ov, t_rel, sig, dur_s,
                              f"Context Overview ({t_rel[0]:.1f} s to +{t_rel[-1]:.1f} s; ~30 s Pre + {dur_s:.2f} s Target + ~30 s Post — Compressed Time Scale)")

        ax_d1 = fig.add_axes([diag_l, 0.305, diag_w, ch_h])
        draw_ecg_grid_panel(ax_d1, t_rel, sig[:, 0], ws, we, dur_s, "Ch1 (ECG)", g0, ylim0, min0, maj0, True, False)
        ax_d1.set_title(f"Enlarged Diagnostic View — Physical Scale: 25.0 mm/s  (Window: [{ws:.2f} s, +{we:.2f} s] enclosing entire target [0.00 s, +{dur_s:.2f} s])",
                        fontsize=8.6, fontweight="bold", loc="left", pad=4, color="#991B1B")

        ax_d2 = fig.add_axes([diag_l, 0.305 - ch_h - 0.012, diag_w, ch_h])
        draw_ecg_grid_panel(ax_d2, t_rel, sig[:, 1], ws, we, dur_s, "Ch2 (ECG)", g1, ylim1, min1, maj1, True, True)
        pdf.savefig(fig)
        plt.close(fig)
        return 1
    else:
        # 2 pages for long targets (> 8.5s):
        # Page 1: Full Context Overview + Diagnostic 25 mm/s Onset Strip [-2.0s .. +8.0s]
        fig1 = new_landscape_fig(f"Page 1 of 2 (Extended Target {dur_s:.2f} s): Full Context (Top) + Onset Diagnostic View [-2.0 s to +8.0 s] at True 25.0 mm/s (Bottom)")
        ax_ov = fig1.add_axes([diag_l, 0.57, diag_w, 0.30])
        render_overview_panel(ax_ov, t_rel, sig, dur_s,
                              f"Context Overview ({t_rel[0]:.1f} s to +{t_rel[-1]:.1f} s; Full Target [{0.0:.2f} s, +{dur_s:.2f} s] Shaded)")

        ax_o1 = fig1.add_axes([diag_l, 0.305, diag_w, ch_h])
        draw_ecg_grid_panel(ax_o1, t_rel, sig[:, 0], -2.0, 8.0, dur_s, "Ch1 (ECG)", g0, ylim0, min0, maj0, True, False)
        ax_o1.set_title("Diagnostic Strip A (Target Onset @ 25.0 mm/s): [-2.00 s pre-onset to +8.00 s into target]",
                        fontsize=8.6, fontweight="bold", loc="left", pad=4, color="#991B1B")
        ax_o2 = fig1.add_axes([diag_l, 0.305 - ch_h - 0.012, diag_w, ch_h])
        draw_ecg_grid_panel(ax_o2, t_rel, sig[:, 1], -2.0, 8.0, dur_s, "Ch2 (ECG)", g1, ylim1, min1, maj1, True, True)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Full Target Overview [0..dur_s] + Diagnostic 25 mm/s Offset Strip [dur_s-8.0 .. dur_s+2.0]
        fig2 = new_landscape_fig(f"Page 2 of 2 (Extended Target {dur_s:.2f} s): Full-Target Interior Strip (Top) + Offset Diagnostic View [{dur_s-8.0:.2f} s to +{dur_s+2.0:.2f} s] at True 25.0 mm/s (Bottom)")
        m_full = (t_rel >= -1.0) & (t_rel <= dur_s + 1.0)
        ax_ft = fig2.add_axes([diag_l, 0.57, diag_w, 0.30])
        render_overview_panel(ax_ft, t_rel[m_full], sig[m_full], dur_s,
                              f"Complete Target-Interval Rhythm Continuity ([-1.0 s to +{dur_s+1.0:.2f} s] — Full Interior View)")

        ws2 = dur_s - 8.0
        we2 = dur_s + 2.0
        ax_e1 = fig2.add_axes([diag_l, 0.305, diag_w, ch_h])
        draw_ecg_grid_panel(ax_e1, t_rel, sig[:, 0], ws2, we2, dur_s, "Ch1 (ECG)", g0, ylim0, min0, maj0, True, False)
        ax_e1.set_title(f"Diagnostic Strip B (Target Offset @ 25.0 mm/s): [{ws2:.2f} s to +{we2:.2f} s]",
                        fontsize=8.6, fontweight="bold", loc="left", pad=4, color="#1E3A8A")
        ax_e2 = fig2.add_axes([diag_l, 0.305 - ch_h - 0.012, diag_w, ch_h])
        draw_ecg_grid_panel(ax_e2, t_rel, sig[:, 1], ws2, we2, dur_s, "Ch2 (ECG)", g1, ylim1, min1, maj1, True, True)
        pdf.savefig(fig2)
        plt.close(fig2)
        return 2


def build_excel_form(manifest_df: pd.DataFrame, out_xlsx: Path):
    """Build 05_EP_Review_Form.xlsx cleanly with dropdown validations and blank verdicts."""
    wb = openpyxl.Workbook()
    ws_inst = wb.active
    ws_inst.title = "Instructions & Progress"
    ws_form = wb.create_sheet("EP Review Form")

    # Styles
    navy = PatternFill("solid", fgColor="1E293B")
    slate = PatternFill("solid", fgColor="F1F5F9")
    amber = PatternFill("solid", fgColor="FEF3C7")
    hdr_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="0F172A")
    bold_font = Font(name="Calibri", size=10, bold=True, color="0F172A")
    reg_font = Font(name="Calibri", size=10, color="1E293B")
    thin = Side(border_style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Instructions sheet
    ws_inst["A1"] = "Blinded ECG Review Form — Target Interruptions in Continuous AF"
    ws_inst["A1"].font = title_font
    inst_lines = [
        ("Reviewer", "Dr. Christopher Cheung"),
        ("Cases Included", "66 blinded strips: 19 Primary Cases (Batch A: CASE-A01..A19, one per patient), 39 Additional Cases (Batch B: CASE-B01..B39), and 8 Reference Controls (CTRL-01..CTRL-08)."),
        ("Suggested Order", "Please review Batch A (19 primary cases) and Reference Controls (8 cases) first, then Batch B (39 cases) as time allows."),
        ("Primary Atrial Rhythm (Col D)", "1. Continued AF throughout the target interval | 2. Organized sinus rhythm | 3. Other identifiable organized atrial rhythm | 4. AF termination or reinitiation within the interval | 5. Atrial rhythm indeterminate"),
        ("Concurrent Ventricular Rhythm (Col E)", "Normally conducted rhythm | Ventricular ectopy | Ventricular tachycardia | Idioventricular rhythm | Bradycardia or pause | Other | Indeterminate"),
        ("Diagnostic Confidence (Col F)", "Low | Moderate | High"),
        ("ECG Interpretability (Col G)", "Good (clear atrial & ventricular signals) | Fair (mild artifact/wander, interpretable) | Poor (limited atrial clarity) | Uninterpretable"),
        ("Transition Timestamps (Col H)", "If selecting Option 4 (within-interval change), enter approximate seconds from 0.00 s target onset (e.g., 'term 1.8s; reinit 4.2s')."),
        ("Review Guidance", "Evaluate atrial electrical activity independently of QRS regularity or ectopy. Do not edit Columns A–C."),
    ]
    r_idx = 3
    for k, v in inst_lines:
        ws_inst.cell(r_idx, 1, k).font = bold_font
        ws_inst.cell(r_idx, 1).fill = slate
        ws_inst.cell(r_idx, 2, v).font = reg_font
        ws_inst.cell(r_idx, 2).alignment = Alignment(wrap_text=True, vertical="top")
        r_idx += 1

    ws_inst.cell(14, 1, "Completion Progress").font = bold_font
    ws_inst.cell(14, 1).fill = amber
    ws_inst.cell(15, 1, "Batch A Completed (/19)").font = bold_font
    ws_inst.cell(15, 2, '=COUNTIFS(\'EP Review Form\'!B2:B67,"Batch A (Primary 19)",\'EP Review Form\'!D2:D67,"<>")')
    ws_inst.cell(16, 1, "Batch B Completed (/39)").font = bold_font
    ws_inst.cell(16, 2, '=COUNTIFS(\'EP Review Form\'!B2:B67,"Batch B (Additional 39)",\'EP Review Form\'!D2:D67,"<>")')
    ws_inst.cell(17, 1, "Controls Completed (/8)").font = bold_font
    ws_inst.cell(17, 2, '=COUNTIFS(\'EP Review Form\'!B2:B67,"Reference Controls (8)",\'EP Review Form\'!D2:D67,"<>")')
    ws_inst.cell(18, 1, "Total Cases Completed (/66)").font = bold_font
    ws_inst.cell(18, 2, '=COUNTA(\'EP Review Form\'!D2:D67)')

    ws_inst.column_dimensions["A"].width = 34
    ws_inst.column_dimensions["B"].width = 105

    # Form sheet
    headers = [
        "Case_ID",
        "Batch",
        "Target_Duration_s",
        "Primary_Atrial_Rhythm_Verdict",
        "Secondary_Concurrent_Ventricular_Rhythm",
        "Diagnostic_Confidence",
        "ECG_Interpretability",
        "Estimated_Transition_Timestamps_s",
        "Reviewer_Notes",
    ]
    for ci, h in enumerate(headers, 1):
        c = ws_form.cell(1, ci, h)
        c.fill = navy
        c.font = hdr_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border
    ws_form.row_dimensions[1].height = 30

    for idx, row in manifest_df.iterrows():
        rr = idx + 2
        ws_form.cell(rr, 1, row["case_id"]).font = bold_font
        ws_form.cell(rr, 2, row["batch_label"]).font = reg_font
        c_dur = ws_form.cell(rr, 3, round(float(row["dur_s"]), 2))
        c_dur.number_format = "0.00"
        c_dur.font = reg_font
        for ci in range(1, 10):
            cell = ws_form.cell(rr, ci)
            cell.border = border
            cell.alignment = Alignment(vertical="center", wrap_text=True, horizontal="center" if ci in (1, 3) else "left")
            if ci <= 3:
                cell.fill = slate
        ws_form.row_dimensions[rr].height = 24

    # Dropdown Data Validations
    dv1 = DataValidation(
        type="list",
        formula1='"1. Continued AF throughout the target interval,2. Organized sinus rhythm,3. Other identifiable organized atrial rhythm,4. AF termination or reinitiation within the interval,5. Atrial rhythm indeterminate"',
        allow_blank=True,
    )
    dv2 = DataValidation(
        type="list",
        formula1='"Normally conducted rhythm,Ventricular ectopy,Ventricular tachycardia,Idioventricular rhythm,Bradycardia or pause,Other,Indeterminate"',
        allow_blank=True,
    )
    dv3 = DataValidation(type="list", formula1='"Low,Moderate,High"', allow_blank=True)
    dv4 = DataValidation(
        type="list",
        formula1='"Good (clear atrial & ventricular signals),Fair (mild artifact/wander, interpretable),Poor (limited atrial clarity),Uninterpretable"',
        allow_blank=True,
    )
    for dv, col in [(dv1, "D"), (dv2, "E"), (dv3, "F"), (dv4, "G")]:
        dv.showInputMessage = True
        ws_form.add_data_validation(dv)
        dv.add(f"{col}2:{col}67")

    widths = [14, 24, 17, 44, 34, 18, 36, 28, 36]
    for ci, w in enumerate(widths, 1):
        ws_form.column_dimensions[get_column_letter(ci)].width = w
    ws_form.freeze_panes = "A2"
    ws_form.auto_filter.ref = "A1:I67"
    wb.save(out_xlsx)


def compute_exact_ltafdb62_completions():
    """Exact 2^10 = 1,024 completions for LTAFDB record 62 matching V8_rev3_partial_id_rigorous.csv."""
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    sub_r = rhy[(rhy["dataset"] == "LTAFDB") & (rhy["record_id"].astype(str) == "62") & (rhy["rhythm_known"])]
    known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
    rem = 43200.0
    clipped_k = []
    for ka, kb in known_ivs:
        if rem <= 1e-9:
            break
        take = min(kb - ka, rem)
        clipped_k.append((ka, ka + take))
        rem -= take
    sub_af = runs[(runs["dataset"] == "LTAFDB") & (runs["record_id"].astype(str) == "62")]
    af_12, T12 = clip_ivs_to_known(merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"]))), clipped_k)
    m_r0 = metrics_from_intervals(af_12, T12)

    u_df = sub_r[
        sub_r["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])
        | ((sub_r["rhythm_label_native"].astype(str) == "SBR") & (sub_r["duration_s"] < 60.0))
    ]
    u_12, _ = clip_ivs_to_known(merge_intervals(list(zip(u_df["start_time_s"], u_df["end_time_s"]))), clipped_k)
    assert len(u_12) == 10
    certain_af = subtract_exclusions(af_12, u_12)
    rows = []
    for idx_c, bits in enumerate(itertools.product([0, 1], repeat=len(u_12))):
        extra_af = [u_12[k] for k in range(len(u_12)) if bits[k] == 1]
        caf = merge_intervals(certain_af + extra_af)
        mc = metrics_from_intervals(caf, T12)
        rows.append({
            "comb_id": idx_c,
            "n_bridged": int(sum(bits)),
            "burden": mc["burden"],
            "density": mc["charitos_density"],
            "is_defined": mc["is_defined"],
            "state": mc["state"],
        })
    df = pd.DataFrame(rows)
    assert len(df) == 1024 and df["is_defined"].sum() == 1023 and (df["state"] == "ALL_AF").sum() == 1
    assert abs(df["density"].min() - 0.163686) < 1e-4 and abs(df["density"].max() - 0.968156) < 1e-4
    return df, m_r0


def build_composite_figure1():
    """Create HRS2027_Figure1.{png,pdf,svg} + _submission.png + source manifest CSV."""
    # Panel A: Representative actual high-impact interruption strip from LTAFDB (Ch1, 25 mm/s, 10 mm/mV; unrevealed ID/tags)
    rep_rid, rep_gs, rep_ge = "49", 20045.156, 20050.312
    rep_dur = rep_ge - rep_gs
    sf = int(round((rep_gs - 2.3) * 128.0))
    st = int(round((rep_gs + 7.7) * 128.0))
    rec = wfdb.rdrecord(str(LTAF_DIR / rep_rid), sampfrom=sf, sampto=st)
    sig_rep = rec.p_signal.astype(np.float64) - np.median(rec.p_signal.astype(np.float64), axis=0, keepdims=True)
    t_rep = (np.arange(sf, st) / 128.0) - rep_gs

    # Panel B: LTAFDB 62 exact 1,024 completions
    df62, m_r0_62 = compute_exact_ltafdb62_completions()

    # Panel C: Verified density-defined coverage & MAE from V8_rev3_fractional_multiscale_summary.csv
    ms = pd.read_csv(RES_DIR / "V8_rev3_fractional_multiscale_summary.csv")
    ms_all = ms[ms["population"] == "All_Defined_For_Rep"].copy()
    scales = [10.0, 30.0, 60.0, 300.0, 900.0, 3600.0]
    scale_lbls = ["10s", "30s", "60s", "5m", "15m", "60m"]
    bin_cov, frac_cov, bin_mae, frac_mae = [], [], [], []
    for h in scales:
        rb = ms_all[(ms_all["h_s"] == h) & (ms_all["representation"] == "Majority_Binary")].iloc[0]
        rf = ms_all[(ms_all["h_s"] == h) & (ms_all["representation"] == "Frac_RightAligned")].iloc[0]
        bin_cov.append(100.0 * rb["n_recs"] / 148.0)
        frac_cov.append(100.0 * rf["n_recs"] / 148.0)
        bin_mae.append(rb["mean_dens_err"])
        frac_mae.append(rf["mean_dens_err"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.9,
    })
    fig = plt.figure(figsize=(11.5, 7.6))
    gs_top = fig.add_axes([0.055, 0.565, 0.915, 0.355])
    gs_bl = fig.add_axes([0.070, 0.085, 0.400, 0.375])
    gs_br = fig.add_axes([0.565, 0.085, 0.405, 0.375])

    # Panel A: Representative 2-Lead ECG Strip (true 10.0s span + 0.5s cal margin; scaled cleanly)
    draw_ecg_grid_panel(
        gs_top, t_rep, sig_rep[:, 0], -2.20, 7.80, rep_dur,
        "Lead 1 (Unfiltered 128 Hz ECG)", 10.0, (-1.15, 1.85), 0.10, 0.50,
        show_cal_pulse=True, draw_xlabel=True,
    )
    gs_top.set_title(
        "A. Transient Non-AF Interruption During Near-Continuous AF (5.16 s Target; 25 mm/s, 10 mm/mV; EP Review Pending)",
        fontsize=10.0, fontweight="bold", loc="left", pad=8, color="#0F172A",
    )

    # Panel B: High-Burden Patient Exhaustive Interval-Assignment Space (1,024 combinations)
    df62_def = df62[df62["is_defined"]].copy()
    x_jit = np.random.default_rng(7).normal(0.0, 0.000018, size=len(df62_def))
    gs_bl.scatter(df62_def["burden"] + x_jit, df62_def["density"],
                  s=11, alpha=0.42, color="#2563EB", edgecolors="none", label="Mixed AF/Non-AF assignments (1,023 / 1,024)")
    dmin_row = df62_def.loc[df62_def["density"].idxmin()]
    dmax_row = df62_def.loc[df62_def["density"].idxmax()]
    gs_bl.scatter([dmin_row["burden"], dmax_row["burden"]], [dmin_row["density"], dmax_row["density"]],
                  s=55, color="#DC2626", marker="o", zorder=6)
    gs_bl.annotate(f"Min = {dmin_row['density']:.4f}", xy=(dmin_row["burden"], dmin_row["density"]),
                   xytext=(0.99945, 0.36), fontsize=7.6, fontweight="bold", color="#991B1B",
                   arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.9))
    gs_bl.annotate(f"Max = {dmax_row['density']:.4f}", xy=(dmax_row["burden"], dmax_row["density"]),
                   xytext=(0.99974, 0.90), fontsize=7.6, fontweight="bold", color="#991B1B",
                   arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.9))
    gs_bl.scatter([m_r0_62["burden"]], [m_r0_62["charitos_density"]],
                  s=75, color="#0F172A", marker="D", zorder=7, label=f"Native annotation (Burden 99.95%, Density {m_r0_62['charitos_density']:.4f})")
    gs_bl.scatter([1.0], [1.03], s=105, color="#DC2626", marker="X", clip_on=False, zorder=7,
                  label="All 10 intervals AF (1 / 1,024: Density undefined)")
    gs_bl.axhline(1.0, color="#94A3B8", lw=0.8, ls="--")
    gs_bl.set_xlim(0.99943, 1.00006)
    gs_bl.set_ylim(0.08, 1.12)
    gs_bl.set_xticks([0.9995, 0.9997, 0.9999, 1.0000])
    gs_bl.set_xticklabels(["99.95%", "99.97%", "99.99%", "100.00%"], fontsize=8.5)
    gs_bl.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    gs_bl.set_yticklabels(["0.20", "0.40", "0.60", "0.80", "1.00"], fontsize=8.5)
    gs_bl.set_xlabel("12-Hour AF Burden (Range: 99.95% to 100.00%)", fontsize=8.8, fontweight="bold")
    gs_bl.set_ylabel("Charitos AF Density (0 to 1)", fontsize=9.0, fontweight="bold")
    gs_bl.set_title("B. AF Density Indeterminacy Under Narrow Burden (10 Brief Interruptions)",
                    fontsize=9.5, fontweight="bold", loc="left", pad=8, color="#0F172A")
    gs_bl.text(0.99945, 0.60, "Exact Density Interval:\n0.1637 to 0.9682 (or Undefined)",
               fontsize=7.5, fontweight="bold", color="#1E3A8A",
               bbox=dict(boxstyle="round,pad=0.24", fc="#EFF6FF", ec="#93C5FD"))
    gs_bl.legend(loc="upper left", fontsize=6.8, frameon=True, framealpha=0.94)
    gs_bl.grid(True, color="#E2E8F0", lw=0.6)

    # Panel C: Density-Defined Coverage (%) + Clinical Callout
    xp = np.arange(len(scale_lbls))
    w = 0.36
    bars_f = gs_br.bar(xp - w / 2, frac_cov, width=w, color="#0D9488", label="Exact % AF Time per Segment (100% defined)")
    bars_b = gs_br.bar(xp + w / 2, bin_cov, width=w, color="#F97316", label="Binary Yes/No AF Call per Segment (92.6% → 58.1%)")
    for b, v in zip(bars_b, bin_cov):
        gs_br.text(b.get_x() + b.get_width() / 2 + 0.03, v + 1.8, f"{v:.1f}%", ha="center", va="bottom",
                   fontsize=7.1, fontweight="bold", color="#9A3412",
                   bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.80))
    for b, v in zip(bars_f, frac_cov):
        gs_br.text(b.get_x() + b.get_width() / 2 - 0.02, v + 1.8, "100%", ha="center", va="bottom",
                   fontsize=7.1, fontweight="bold", color="#115E59")
    gs_br.set_xticks(xp)
    gs_br.set_xticklabels(scale_lbls, fontsize=8.8)
    gs_br.set_ylim(0, 134)
    gs_br.set_yticks([0, 20, 40, 60, 80, 100])
    gs_br.set_xlabel("ECG Segment Duration (n = 148 Mixed Multi-Hour Recordings)", fontsize=8.6, fontweight="bold")
    gs_br.set_ylabel("Recordings With Defined AF Density (%)", fontsize=9.0, fontweight="bold")
    gs_br.set_title("C. AF Density Measurability by Segment Reporting Rule",
                    fontsize=9.8, fontweight="bold", loc="left", pad=8, color="#0F172A")
    gs_br.text(0.03, 0.09,
               "Binary yes/no calls erase short non-AF breaks,\nleaving density undefined in up to 41.9% of recordings\n(10-s mean |error|: 0.00001 vs 0.01038)",
               transform=gs_br.transAxes, fontsize=7.0, color="#0F172A",
               bbox=dict(boxstyle="round,pad=0.25", fc="#F0FDFA", ec="#5EEAD4", alpha=0.95))
    gs_br.legend(loc="upper center", fontsize=7.0, frameon=True, framealpha=0.95)
    gs_br.grid(axis="y", color="#E2E8F0", lw=0.6)

    fig_png = OUT_ROOT / "HRS2027_Figure1.png"
    fig_sub = OUT_ROOT / "HRS2027_Figure1_submission.png"
    fig_pdf = OUT_ROOT / "HRS2027_Figure1.pdf"
    fig_svg = OUT_ROOT / "HRS2027_Figure1.svg"
    fig.savefig(fig_png, dpi=600)
    fig.savefig(fig_pdf)
    fig.savefig(fig_svg)
    plt.close(fig)

    # Compact HRS submission PNG (1800px wide RGB)
    with Image.open(fig_png) as im:
        ratio = 1800.0 / im.width
        im_sub = im.convert("RGB").resize((1800, int(round(im.height * ratio))), Image.Resampling.LANCZOS)
        im_sub.save(fig_sub, format="PNG", optimize=True)

    # Exact Figure 1 source-data manifest
    f_manifest = pd.DataFrame([
        {"panel": "Panel A", "description": "Representative raw WFDB ECG strip (5.16s target + context), unrevealed Case_ID & blinded pre-review",
         "source_artifact": "LTAFDB WFDB rec 49 (20045.156-20050.312s) via V8_rev3_ep_review_packet_top60.csv",
         "verification_summary": "128.0 Hz Ch1 physical mV, median DC offset removed only, 25 mm/s x 10 mm/mV"},
        {"panel": "Panel B", "description": "LTAFDB record 62 exhaustive 2^10 = 1,024 whole-interval completion space under R2_VT_SBR60_Unknown",
         "source_artifact": "experiments/af_density_v8/results/V8_rev3_partial_id_rigorous.csv",
         "verification_summary": "Certified burden [0.999500, 1.000000]; 1,023 MIXED [0.1637, 0.9682] + 1 ALL_AF (undefined); R0 D=0.1655"},
        {"panel": "Panel C", "description": "Density-defined coverage (%) across 10s-60m for Majority Binary vs Fractional Occupancy Oracles",
         "source_artifact": "experiments/af_density_v8/results/V8_rev3_fractional_multiscale_summary.csv",
         "verification_summary": "n=148 native mixed multi-hour recordings; Binary 137/148 (92.6%) -> 86/148 (58.1%); Fractional 148/148 (100.0%)"},
    ])
    f_manifest.to_csv(OUT_ROOT / "HRS2027_Figure1_source_manifest.csv", index=False)


def main():
    EP_DIR.mkdir(parents=True, exist_ok=True)
    INV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Freezing Selection & Generating Blinding Manifests ===")
    full_df = build_frozen_selection()

    # Save Case Selection Manifest (all selection rules & timestamps, investigator-only)
    full_df.to_csv(INV_DIR / "case_selection_manifest.csv", index=False)
    map_cols = [
        "case_id", "batch_code", "batch_label", "presentation_order", "wfdb_record_id",
        "patient_id", "gap_start_s", "gap_end_s", "dur_s", "native_tags",
        "is_native_ventricular", "r1_class", "r0_density", "single_gap_abs_delta_D",
        "pct_of_total_r0_nonaf", "control_truth", "selection_freeze_utc",
    ]
    full_df[map_cols].to_csv(INV_DIR / "case_mapping.csv", index=False)

    print("=== Step 2: Generating Blinded ECG Strip PDFs & Hashing Source Waveforms ===")
    hash_rows = []
    batch_pdfs = [
        ("BATCH_A", EP_DIR / "02_Primary_19_Cases.pdf"),
        ("BATCH_B", EP_DIR / "03_Additional_39_Cases.pdf"),
        ("CONTROL", EP_DIR / "04_Control_Cases.pdf"),
    ]
    total_pages = {}
    for bcode, pdf_path in batch_pdfs:
        sub = full_df[full_df["batch_code"] == bcode].sort_values("presentation_order").reset_index(drop=True)
        pages = 0
        with PdfPages(pdf_path) as pdf:
            for i, row in sub.iterrows():
                t_rel, sig, fs, sig_names, hinfo = extract_case_waveform(row)
                hash_rows.append(hinfo)
                pages += render_case_pages(pdf, row, t_rel, sig, fs, i + 1, len(sub))
        total_pages[bcode] = pages
        print(f"  Wrote {pdf_path} ({len(sub)} cases, {pages} PDF pages)")

    pd.DataFrame(hash_rows).to_csv(INV_DIR / "source_waveform_hashes.csv", index=False)

    print("=== Step 3: Building Blank Blinded EP Review Form (.xlsx) ===")
    build_excel_form(full_df.reset_index(drop=True), EP_DIR / "05_EP_Review_Form.xlsx")

    print("=== Step 4: Building HRS 2027 Composite Figure 1 ===")
    build_composite_figure1()
    print("Done all binary/data assets. Total pages:", total_pages)


if __name__ == "__main__":
    main()
