#!/usr/bin/env python3
"""Pipeline 10-Epoch Convergence Panel, Conditional 15E Extension & Finalist Selector (Protocol of Record v3.2):
1. Ingests Lead-I & Lead-II 3E screening results from refine-logs/wavelet_ssl_1110000/full/queue.sqlite.
2. Executes pre-specified immutable 12-model panel (24 runs, 240 epochs total, saving every epoch).
3. Evaluates terminal learning dynamics: M_max, e_max, M_10, and terminal regression slope beta_{7:10}.
4. Triggers conditional 15-epoch extension with noise floor: e_max in {9, 10} AND (delta_10:8 > 0.0010 OR beta_{7:10} > 0.0005).
5. Predefines single checkpoint rule: e_m* = argmax_e r_val(e) to extract all metrics synchronously.
6. Computes hardened robust composite score S_m on eligible models E (excluding diagnostic negative control):
   S_m = z_robust(r_mean) + 0.5*z_robust(r05_mean) + 0.5*z_robust(P_mean) + 0.5*z_robust(T_mean) + 0.25*z_robust(r_min)
   (with MAD -> IQR -> zero discriminative weight fallbacks).
7. Sequentially selects 4 unique finalists with collision handling for downstream objective and spatial development.
"""

from __future__ import annotations

import argparse, json, os, sqlite3, sys, time
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import atomic_json, run_queue

SCREENING_DB = ROOT / "refine-logs/wavelet_ssl_1110000/full/queue.sqlite"
CONVERGENCE_DIR = ROOT / "refine-logs/convergence_10e"
MANIFEST_PATH = CONVERGENCE_DIR / "manifest.json"
QUEUE_DB_PATH = CONVERGENCE_DIR / "queue.sqlite"

# Pre-specified immutable 12-model mechanistic panel (informed by screening results)
FROZEN_CONVERGENCE_PANEL = [
    "conv_control",                                # CNN Baseline representative
    "R7_morlet_mag_ueg_real",                      # UEG Real-component representative
    "R5_morlet_mag_ueg_phase_wyatt",               # UEG Wyatt-phase representative
    "ssl_log_magnitude_phase_sin_local_gated_add", # Local-scale Gated SSL representative
    "ssl_log_magnitude_real_both_gated_add",       # Dual-view Gated SSL representative
    "tf_sc16_cy4",                                 # Low-scale TimeSformer representative (sc16 cy4)
    "tf_sc16_cy8",                                 # High-cycle TimeSformer representative (sc16 cy8)
    "del_wave_ce",                                 # Delineation-auxiliary representative
    "C1_E1_morlet_mag_morlet_phase",               # Disentangled core-matrix representative
    "A0_raw",                                      # Raw waveform baseline control
    "A0_wave_noSSL_gated_add",                     # Wavelet no-SSL gated control
    "ssl_magnitude_phase_both_cross_attn",         # Diagnostic negative control (Cross-attention failure mode)
]

NEGATIVE_CONTROLS = {"ssl_magnitude_phase_both_cross_attn"}

# Architectural family tags for diversity selection
ARCH_FAMILIES = {
    "conv_control": "cnn_baseline",
    "R7_morlet_mag_ueg_real": "ueg_representation",
    "R5_morlet_mag_ueg_phase_wyatt": "ueg_representation",
    "ssl_log_magnitude_phase_sin_local_gated_add": "gated_ssl",
    "ssl_log_magnitude_real_both_gated_add": "gated_ssl",
    "tf_sc16_cy4": "timesformer",
    "tf_sc16_cy8": "timesformer",
    "del_wave_ce": "delineation_aux",
    "C1_E1_morlet_mag_morlet_phase": "core_matrix",
    "A0_raw": "raw_control",
    "A0_wave_noSSL_gated_add": "wavelet_control",
    "ssl_magnitude_phase_both_cross_attn": "negative_control",
}

def load_screening_results(db_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not db_path.is_file():
        raise FileNotFoundError(f"Screening DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, status, command_json, summary_json FROM jobs").fetchall()
    conn.close()

    l0_results = {}
    l1_results = {}
    for jid, status, cmd_str, sum_str in rows:
        if status != "completed" or not sum_str:
            continue
        summary = json.loads(sum_str)
        cmd = json.loads(cmd_str) if cmd_str else []
        if "_l0" in jid:
            clean = jid.replace("_1110000_s42_l0", "")
            l0_results[clean] = {"id": jid, "cmd": cmd, "summary": summary}
        elif "_l1" in jid:
            clean = jid.replace("_1110000_s42_l1", "")
            l1_results[clean] = {"id": jid, "cmd": cmd, "summary": summary}

    return l0_results, l1_results

def build_convergence_manifest(panel_archs: list[str], l0: dict[str, Any], l1: dict[str, Any], target_epochs: int = 10, prefix: str = "conv10e") -> dict[str, Any]:
    CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
    cells = []

    for arch in panel_archs:
        for lead_idx, lead_dict, lead_tag in [(0, l0, "l0"), (1, l1, "l1")]:
            orig = lead_dict.get(arch)
            if not orig:
                counterpart = l0.get(arch) if lead_idx == 1 else l1.get(arch)
                if not counterpart:
                    continue
                cmd = list(counterpart["cmd"])
                old_tag = "l0" if lead_idx == 1 else "l1"
                new_cmd = []
                for i, token in enumerate(cmd):
                    if token == "--observed-leads":
                        new_cmd.append(token)
                    elif i > 0 and cmd[i-1] == "--observed-leads":
                        new_cmd.append(str(lead_idx + 1))
                    elif token.endswith(f"_{old_tag}"):
                        new_cmd.append(token.replace(f"_{old_tag}", f"_{lead_tag}"))
                    elif f"/{old_tag}" in token or f"_{old_tag}/" in token:
                        new_cmd.append(token.replace(f"_{old_tag}", f"_{lead_tag}"))
                    else:
                        new_cmd.append(token)
                cmd = new_cmd
            else:
                cmd = list(orig["cmd"])

            run_id = f"{prefix}_{arch}_s42_{lead_tag}"
            run_dir = CONVERGENCE_DIR / "runs" / run_id

            modified_cmd = []
            skip_next = False
            for idx, token in enumerate(cmd):
                if skip_next:
                    skip_next = False
                    continue
                if token == "--epochs":
                    modified_cmd.extend(["--epochs", str(target_epochs)])
                    skip_next = True
                elif token == "--run-name":
                    modified_cmd.extend(["--run-name", run_id])
                    skip_next = True
                elif token == "--output-dir":
                    modified_cmd.extend(["--output-dir", str(run_dir)])
                    skip_next = True
                elif token == "--checkpoint-policy":
                    modified_cmd.extend(["--checkpoint-policy", "every_epoch"])
                    skip_next = True
                else:
                    modified_cmd.append(token)

            if "--checkpoint-policy" not in modified_cmd:
                modified_cmd.extend(["--checkpoint-policy", "every_epoch"])

            cells.append({
                "id": run_id,
                "architecture": arch,
                "lead": lead_idx,
                "epochs": target_epochs,
                "seed": 42,
                "output_dir": str(run_dir),
                "command": modified_cmd,
            })

    manifest = {
        "version": 1,
        "experiment_name": f"convergence_{target_epochs}epoch_panel",
        "description": f"{target_epochs}-Epoch Convergence Study for Frozen 12-Model Panel across Lead I & Lead II",
        "total_jobs": len(cells),
        "total_epochs": len(cells) * target_epochs,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "panel_architectures": panel_archs,
        "cells": cells,
    }
    atomic_json(MANIFEST_PATH, manifest)
    return manifest

def init_sqlite_queue(manifest: dict[str, Any]) -> None:
    QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QUEUE_DB_PATH)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                ordinal INTEGER,
                command_json TEXT,
                command_sha256 TEXT,
                cell_json TEXT,
                status TEXT,
                attempts INTEGER DEFAULT 0,
                child_pid INTEGER,
                started_at TEXT,
                completed_at TEXT,
                returncode INTEGER,
                error TEXT,
                log_path TEXT,
                output_dir TEXT,
                summary_json TEXT,
                updated_at TEXT
            )
        """)
        for idx, cell in enumerate(manifest["cells"]):
            cmd_json = json.dumps(cell["command"])
            cell_json = json.dumps(cell)
            conn.execute("""
                INSERT INTO jobs (id, ordinal, command_json, cell_json, status, output_dir, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(id) DO UPDATE SET command_json=excluded.command_json, cell_json=excluded.cell_json
            """, (cell["id"], idx, cmd_json, cell_json, cell["output_dir"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    conn.close()

def compute_terminal_slope(series: list[float], epochs: list[int] = None) -> float:
    """Computes least-squares linear regression slope beta over terminal window (e.g. epochs 7..10)."""
    if len(series) < 2:
        return 0.0
    if epochs is None:
        epochs = list(range(len(series)))
    x = np.array(epochs, dtype=float)
    y = np.array(series, dtype=float)
    if np.all(y == y[0]):
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

def robust_zscore(values: list[float], eps: float = 1e-6) -> np.ndarray:
    """Hardened robust z-score:
    1. Uses (x - median) / (1.4826 * MAD)
    2. Fallback to IQR if MAD < eps
    3. Zeroes out discriminative weight if both MAD and IQR < eps
    """
    arr = np.array(values, dtype=float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad > eps:
        return (arr - med) / (1.4826 * mad)
    
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25
    if iqr > eps:
        return (arr - med) / (0.7413 * iqr)
    
    return np.zeros_like(arr)

def analyze_convergence_dynamics(conv_dir: Path) -> list[dict[str, Any]]:
    """Extracts trajectory curves and evaluates M_max, e_max, M_10, and terminal regression slope beta_{7:10}
    using a single predefined checkpoint rule: e* = argmax_e r_val(e).
    """
    runs_dir = conv_dir / "runs"
    if not runs_dir.is_dir():
        print("No convergence runs directory found.")
        return []

    results = []
    for run in sorted(runs_dir.iterdir()):
        metrics_file = run / "metrics.jsonl"
        if not metrics_file.is_file():
            continue

        epochs_data = []
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    epochs_data.append(json.loads(line))

        if len(epochs_data) < 2:
            continue

        r_curve = [ep.get("val_missing_pearson", 0.0) for ep in epochs_data]
        p_curve = [ep.get("P_iou", 0.0) for ep in epochs_data]
        t_curve = [ep.get("T_iou", 0.0) for ep in epochs_data]
        qrs_curve = [ep.get("QRS_iou", 0.0) for ep in epochs_data]
        bf1_curve = [ep.get("boundary_f1_smoke", 0.0) for ep in epochs_data]
        r05_curve = [ep.get("val_missing_pearson_p05", 0.0) for ep in epochs_data]

        best_e_idx = int(np.argmax(r_curve))
        best_e = best_e_idx + 1

        r_star = float(r_curve[best_e_idx])
        r05_star = float(r05_curve[best_e_idx])
        p_star = float(p_curve[best_e_idx])
        t_star = float(t_curve[best_e_idx])
        qrs_star = float(qrs_curve[best_e_idx])
        bf1_star = float(bf1_curve[best_e_idx])

        r_10 = float(r_curve[-1])
        r_8 = float(r_curve[-3]) if len(r_curve) >= 3 else float(r_curve[0])
        delta_10_8 = r_10 - r_8

        n_term = min(4, len(r_curve))
        term_r = r_curve[-n_term:]
        term_epochs = list(range(len(r_curve) - n_term + 1, len(r_curve) + 1))
        beta_7_10 = compute_terminal_slope(term_r, term_epochs)

        results.append({
            "run_id": run.name,
            "architecture": run.name.replace("conv10e_", "").replace("conv15e_", "").replace("_s42_l0", "").replace("_s42_l1", ""),
            "lead": 0 if "_l0" in run.name else 1,
            "epochs_completed": len(epochs_data),
            "e_star": best_e,
            "r_star": r_star,
            "r05_star": r05_star,
            "p_star": p_star,
            "t_star": t_star,
            "qrs_star": qrs_star,
            "bf1_star": bf1_star,
            "r_10": r_10,
            "delta_10_8": delta_10_8,
            "beta_7_10": beta_7_10,
            "r_trajectory": r_curve,
        })

    print(f"\n" + "="*88)
    print(f"  CONVERGENCE DYNAMICS: Synchronous Checkpoint e*, M_10, beta_{{7:10}}, and Delta_{{10:8}}")
    print("="*88)
    for res in sorted(results, key=lambda x: x["r_star"], reverse=True):
        curve_str = ", ".join([f"{x:.3f}" for x in res["r_trajectory"][:10]])
        print(f"[{res['run_id']:<42}] r*={res['r_star']:.4f} (ep {res['e_star']:02d}) | beta_7:10={res['beta_7_10']:+.5f}/ep | d10-8={res['delta_10_8']:+.4f} | P*={res['p_star']:.3f} | T*={res['t_star']:.3f}")
        print(f"   Trajectory: [{curve_str}]")

    return results

def check_conditional_15e_extension(convergence_results: list[dict[str, Any]]) -> list[str]:
    """Identifies candidate models with meaningful learning momentum:
    e_max in {9, 10} AND (delta_10_8 > 0.0010 OR beta_7:10 > 0.0005).
    """
    archs_to_extend = []
    for r in convergence_results:
        if r["architecture"] in NEGATIVE_CONTROLS:
            continue
        if r["e_star"] >= 9 and (r["delta_10_8"] > 0.0010 or r["beta_7_10"] > 0.0005):
            archs_to_extend.append(r["architecture"])

    unique_extend = list(dict.fromkeys(archs_to_extend))
    if unique_extend:
        print("\n" + "="*80)
        print(f"  CONDITIONAL 15-EPOCH EXTENSION TRIGGERED FOR {len(unique_extend)} MODELS (Exceeds Noise Floor)")
        print("="*80)
        for a in unique_extend:
            print(f"  -> Model '{a}' verified climbing (delta_10:8 > 0.001 or beta > 0.0005); extending to 15 epochs.")
    else:
        print("\n  -> Prespecified Convergence Criterion Satisfied across all models (no residual momentum).")
    return unique_extend

def compute_finalist_selection(convergence_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculates robust composite selection score S_m over eligible models E (excluding negative control):
    S_m = z_robust(r_mean) + 0.5*z_robust(r05_mean) + 0.5*z_robust(P_mean) + 0.5*z_robust(T_mean) + 0.25*z_robust(r_min)
    and executes collision-free sequential selection for the 4 finalist slots.
    """
    archs = sorted(list(set(r["architecture"] for r in convergence_results)))
    eligible_archs = [a for a in archs if a not in NEGATIVE_CONTROLS]

    arch_metrics = {}
    for arch in eligible_archs:
        l0 = next((r for r in convergence_results if r["architecture"] == arch and r["lead"] == 0), None)
        l1 = next((r for r in convergence_results if r["architecture"] == arch and r["lead"] == 1), None)
        if not l0 or not l1:
            continue
        
        r_mean = (l0["r_star"] + l1["r_star"]) / 2.0
        r_min = min(l0["r_star"], l1["r_star"])
        r_delta = abs(l0["r_star"] - l1["r_star"])
        r05_mean = (l0["r05_star"] + l1["r05_star"]) / 2.0
        p_mean = (l0["p_star"] + l1["p_star"]) / 2.0
        t_mean = (l0["t_star"] + l1["t_star"]) / 2.0
        pt_morph_sum = p_mean + t_mean
        family = ARCH_FAMILIES.get(arch, "other")

        arch_metrics[arch] = {
            "arch": arch,
            "family": family,
            "r_mean": r_mean,
            "r_min": r_min,
            "r_delta": r_delta,
            "r05_mean": r05_mean,
            "p_mean": p_mean,
            "t_mean": t_mean,
            "pt_morph_sum": pt_morph_sum,
        }

    if not arch_metrics:
        print("Insufficient paired Lead-I/Lead-II runs to compute finalist selection.")
        return []

    names = list(arch_metrics.keys())
    z_r = robust_zscore([arch_metrics[n]["r_mean"] for n in names])
    z_rmin = robust_zscore([arch_metrics[n]["r_min"] for n in names])
    z_r05 = robust_zscore([arch_metrics[n]["r05_mean"] for n in names])
    z_p = robust_zscore([arch_metrics[n]["p_mean"] for n in names])
    z_t = robust_zscore([arch_metrics[n]["t_mean"] for n in names])

    scored_records = []
    for i, name in enumerate(names):
        s_m = z_r[i] + 0.5 * z_r05[i] + 0.5 * z_p[i] + 0.5 * z_t[i] + 0.25 * z_rmin[i]
        arch_metrics[name]["score_S_m"] = float(s_m)
        scored_records.append(arch_metrics[name])

    scored_records.sort(key=lambda x: x["score_S_m"], reverse=True)

    print("\n" + "="*88)
    print("  ROBUST COMPOSITE SELECTION LEADERBOARD S_m (OVER ELIGIBLE SET E)")
    print("="*88)
    for r in scored_records:
        print(f"Arch: {r['arch']:<38} | S_m: {r['score_S_m']:+6.3f} | r_mean: {r['r_mean']:.4f} | r_min: {r['r_min']:.4f} | r05: {r['r05_mean']:.4f} | P: {r['p_mean']:.3f} | T: {r['t_mean']:.3f}")

    # Sequential collision-free finalist slot selection:
    finalists = []
    selected_names = set()

    # Slot 1: Highest composite performance S_m
    slot1 = scored_records[0]
    finalists.append(("Slot 1 (Highest Composite S_m)", slot1))
    selected_names.add(slot1["arch"])

    # Slot 2: Highest P/T morphology not already selected
    rem_pt = [r for r in scored_records if r["arch"] not in selected_names]
    rem_pt.sort(key=lambda x: x["pt_morph_sum"], reverse=True)
    slot2 = rem_pt[0]
    finalists.append(("Slot 2 (Highest P/T Morphology)", slot2))
    selected_names.add(slot2["arch"])

    # Slot 3: Highest S_m alternative architectural family not already selected
    slot1_family = slot1["family"]
    rem_fam = [r for r in scored_records if r["arch"] not in selected_names and r["family"] != slot1_family]
    slot3 = rem_fam[0] if rem_fam else [r for r in scored_records if r["arch"] not in selected_names][0]
    finalists.append(("Slot 3 (Best Alternative Family)", slot3))
    selected_names.add(slot3["arch"])

    # Slot 4: Predefined Baseline / Control not already selected
    control_candidates = ["conv_control", "A0_raw", "A0_wave_noSSL_gated_add"]
    slot4_arch = next((c for c in control_candidates if c not in selected_names), None)
    slot4 = next((r for r in scored_records if r["arch"] == slot4_arch), None)
    if not slot4:
        slot4 = [r for r in scored_records if r["arch"] not in selected_names][0]
    finalists.append(("Slot 4 (Baseline / Control)", slot4))
    selected_names.add(slot4["arch"])

    print("\n" + "="*88)
    print("  FINAL 4 CONVERGED BACKBONE FINALISTS (COLLISION-FREE SELECTION)")
    print("="*88)
    for slot_name, f in finalists:
        print(f"  * {slot_name:<35} -> {f['arch']:<35} (Family: {f['family']}, S_m: {f['score_S_m']:+.3f}, r_mean: {f['r_mean']:.4f})")

    return [f[1] for f in finalists]

def main():
    parser = argparse.ArgumentParser(description="10-Epoch Convergence Pipeline & Finalist Selector (v3.2)")
    parser.add_argument("--auto-launch", action="store_true", help="Launch the queue immediately after generation")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze existing 10E convergence curves")
    args = parser.parse_args()

    if args.analyze_only:
        conv_res = analyze_convergence_dynamics(CONVERGENCE_DIR)
        check_conditional_15e_extension(conv_res)
        compute_finalist_selection(conv_res)
        return

    print("="*88)
    print("  LAUNCHING PRE-SPECIFIED 12-MODEL IMMUTABLE CONVERGENCE PANEL (v3.2)")
    print("="*88)

    l0, l1 = load_screening_results(SCREENING_DB)
    print(f"Loaded screening state: Lead-I completed = {len(l0)}, Lead-II completed = {len(l1)}")

    for idx, arch in enumerate(FROZEN_CONVERGENCE_PANEL):
        print(f"  [{idx+1:02d}/12] Pre-specified panel member: {arch}")

    manifest = build_convergence_manifest(FROZEN_CONVERGENCE_PANEL, l0, l1, target_epochs=10, prefix="conv10e")
    init_sqlite_queue(manifest)
    print(f"\nManifest written to: {MANIFEST_PATH}")
    print(f"SQLite Queue initialized: {QUEUE_DB_PATH} ({manifest['total_jobs']} jobs, {manifest['total_epochs']} total epochs)")

    if args.auto_launch:
        print("\nExecuting 10-Epoch Convergence Panel on GPU...")
        code = run_queue(
            MANIFEST_PATH, project_root=ROOT, max_attempts=2, min_free_gib=8,
            min_available_ram_gib=5, continue_on_error=True, max_consecutive_failures=2,
            max_total_failures=5, max_gpu_used_mib=1024,
            resource_timeout_seconds=21600, job_timeout_seconds=14400,
        )
        print(f"10-Epoch convergence queue exited with code {code}")
        
        conv_res = analyze_convergence_dynamics(CONVERGENCE_DIR)
        extend_archs = check_conditional_15e_extension(conv_res)
        if extend_archs:
            print(f"\nExtending {len(extend_archs)} models to 15 epochs...")
            ext_manifest = build_convergence_manifest(extend_archs, l0, l1, target_epochs=15, prefix="conv15e")
            init_sqlite_queue(ext_manifest)
            run_queue(
                MANIFEST_PATH, project_root=ROOT, max_attempts=2, min_free_gib=8,
                min_available_ram_gib=5, continue_on_error=True, max_consecutive_failures=2,
                max_total_failures=5, max_gpu_used_mib=1024,
                resource_timeout_seconds=21600, job_timeout_seconds=14400,
            )
            conv_res = analyze_convergence_dynamics(CONVERGENCE_DIR)

        compute_finalist_selection(conv_res)

if __name__ == "__main__":
    main()
