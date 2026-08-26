#!/usr/bin/env python3
"""Focused 8-Job 1111002 Objective Probe with Matched Convergence Rule:
Tests the 1111002 factorial mask (repolarization loss + complex geometry) on the 4 converged finalists
across Lead I & Lead II at 10 epochs (4 archs x 2 leads = 8 jobs), with conditional 15E extension if climbing.
Outputs the definitive Winner of (Backbone x Objective) to anchor the Spatial Search.
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

PROBE_DIR = ROOT / "refine-logs/probe_1111002_8job"
MANIFEST_PATH = PROBE_DIR / "manifest.json"
QUEUE_DB_PATH = PROBE_DIR / "queue.sqlite"

def build_probe_manifest(finalists: list[str], target_epochs: int = 10, prefix: str = "probe1111002") -> dict[str, Any]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    conv_manifest = ROOT / "refine-logs/convergence_10e/manifest.json"
    if not conv_manifest.is_file():
        raise FileNotFoundError(f"Convergence manifest not found: {conv_manifest}")
    with open(conv_manifest) as f:
        conv_data = json.load(f)

    cmd_map = {}
    for cell in conv_data["cells"]:
        cmd_map[(cell["architecture"], cell["lead"])] = list(cell["command"])

    cells = []
    for arch in finalists:
        for lead_idx in [0, 1]:
            lead_tag = f"l{lead_idx}"
            template = cmd_map.get((arch, lead_idx))
            if not template:
                print(f"Warning: Template for ({arch}, {lead_idx}) not found in convergence manifest.")
                continue

            run_id = f"{prefix}_{arch}_s42_{lead_tag}"
            run_dir = PROBE_DIR / "runs" / run_id

            cmd = []
            skip_next = False
            for token in template:
                if skip_next:
                    skip_next = False
                    continue
                if token == "--factorial-mask":
                    cmd.extend(["--factorial-mask", "1111002"])
                    skip_next = True
                elif token == "--epochs":
                    cmd.extend(["--epochs", str(target_epochs)])
                    skip_next = True
                elif token == "--run-name":
                    cmd.extend(["--run-name", run_id])
                    skip_next = True
                elif token == "--output-dir":
                    cmd.extend(["--output-dir", str(run_dir)])
                    skip_next = True
                else:
                    cmd.append(token)

            cells.append({
                "id": run_id,
                "architecture": arch,
                "lead": lead_idx,
                "factorial_mask": "1111002",
                "epochs": target_epochs,
                "seed": 42,
                "output_dir": str(run_dir),
                "command": cmd
            })

    manifest = {
        "version": 1,
        "experiment_name": f"probe_1111002_{target_epochs}epoch",
        "description": f"Targeted probe testing factorial mask 1111002 on 4 finalists at {target_epochs} epochs",
        "total_jobs": len(cells),
        "total_epochs": len(cells) * target_epochs,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finalists": finalists,
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

def analyze_probe_vs_baseline(finalists: list[str]) -> dict[str, Any]:
    """Compares 1111002 probe against 1110000 convergence runs at matched checkpoints."""
    conv_dir = ROOT / "refine-logs/convergence_10e/runs"
    probe_runs_dir = PROBE_DIR / "runs"
    
    print("\n" + "="*85)
    print("  1111002 REPOLARIZATION PROBE VS 1110000 BASELINE (MATCHED CHECKPOINTS)")
    print("="*85)

    comparison = []
    for arch in finalists:
        for lead in [0, 1]:
            lead_tag = f"l{lead}"
            # Load 1110000 baseline
            b_file = conv_dir / f"conv10e_{arch}_s42_{lead_tag}/summary.json"
            p_file = probe_runs_dir / f"probe1111002_{arch}_s42_{lead_tag}/summary.json"
            if not b_file.is_file() or not p_file.is_file():
                continue
            with open(b_file) as f: b = json.load(f)
            with open(p_file) as f: p = json.load(f)

            delta_r = p.get("val_missing_pearson", 0) - b.get("val_missing_pearson", 0)
            delta_t = p.get("T_iou", 0) - b.get("T_iou", 0)
            delta_p = p.get("P_iou", 0) - b.get("P_iou", 0)
            delta_r05 = p.get("val_missing_pearson_p05", 0) - b.get("val_missing_pearson_p05", 0)

            comparison.append({
                "arch": arch,
                "lead": lead,
                "b_r": b.get("val_missing_pearson"),
                "p_r": p.get("val_missing_pearson"),
                "delta_r": delta_r,
                "b_t": b.get("T_iou"),
                "p_t": p.get("T_iou"),
                "delta_t": delta_t,
                "delta_r05": delta_r05
            })

            print(f"[{arch:<32} {lead_tag}] Delta r: {delta_r:+.4f} (1110:{b.get('val_missing_pearson'):.4f} -> 1111:{p.get('val_missing_pearson'):.4f}) | Delta T_iou: {delta_t:+.4f} | Delta r05: {delta_r05:+.4f}")

    return {"comparison": comparison}

def main():
    parser = argparse.ArgumentParser(description="Focused 8-Job 1111002 Probe with Matched Convergence Rule")
    parser.add_argument("--finalists", nargs="+", help="Explicit list of 4 finalists (otherwise auto-selected)")
    parser.add_argument("--auto-launch", action="store_true", help="Launch the queue immediately after generation")
    args = parser.parse_args()

    if args.finalists:
        finalists = args.finalists
    else:
        finalists = ["conv_control", "R7_morlet_mag_ueg_real", "ssl_log_magnitude_phase_sin_local_gated_add", "A0_raw"]

    print("="*80)
    print("  BUILDING 8-JOB 1111002 OBJECTIVE PROBE QUEUE (10 EPOCHS)")
    print(f"  Finalists: {finalists}")
    print("="*80)

    manifest = build_probe_manifest(finalists, target_epochs=10, prefix="probe1111002")
    init_sqlite_queue(manifest)
    print(f"Probe manifest written to: {MANIFEST_PATH}")
    print(f"Probe queue initialized: {QUEUE_DB_PATH}")

    if args.auto_launch:
        print("\nLaunching 8-Job 1111002 Probe on GPU...")
        code = run_queue(
            MANIFEST_PATH, project_root=ROOT, max_attempts=2, min_free_gib=8,
            min_available_ram_gib=5, continue_on_error=True, max_consecutive_failures=2,
            max_total_failures=5, max_gpu_used_mib=1024,
            resource_timeout_seconds=21600, job_timeout_seconds=14400,
        )
        print(f"1111002 probe queue exited with code {code}")
        analyze_probe_vs_baseline(finalists)

if __name__ == "__main__":
    main()
