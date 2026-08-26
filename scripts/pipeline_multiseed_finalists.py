#!/usr/bin/env python3
"""Pipeline Definitive Multi-Seed Confirmation & Tri-Realization Evaluation (Protocol of Record v3.3):
1. Ingests pre-frozen validation candidate ranking list [m_1, m_2, m_3, ...].
2. Trains/evaluates all 3 seed realizations (theta_42, theta_43, theta_44) for candidate m_1.
3. Applies prespecified engineering/optimization stability gate:
   range_seed(r) <= 0.025 AND SD_seed(r) <= 0.015.
   If failed: deterministically advances to m_2 in the frozen fallback list.
4. Designates the PRIMARY CONFIGURATION (the architecture family, not a cherry-picked seed).
5. Freezes all 3 realizations (theta_42, theta_43, theta_44) and evaluates each independently on:
   - PTB-XL Internal Test Fold (reporting mean +- SD_seed)
   - External RDB Blinded Transfer Set (reporting mean +- SD_seed)
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

MULTISEED_DIR = ROOT / "refine-logs/multiseed_final_confirmation"
MANIFEST_PATH = MULTISEED_DIR / "manifest.json"
QUEUE_DB_PATH = MULTISEED_DIR / "queue.sqlite"
FALLBACK_RECORD_PATH = MULTISEED_DIR / "frozen_fallback_ranking.json"

# Prespecified Engineering / Optimization Stability Gate Thresholds
MAX_ALLOWED_SEED_RANGE = 0.025
MAX_ALLOWED_SEED_STD = 0.015

def record_frozen_fallback_list(ranked_candidates: list[dict[str, Any]]) -> None:
    """Permanently records the deterministic fallback list m_1, m_2, m_3, ... before seed execution."""
    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(FALLBACK_RECORD_PATH, {
        "version": 1,
        "purpose": "frozen_deterministic_fallback_ranking",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates": ranked_candidates
    })

def build_multiseed_manifest(selected_model: dict[str, Any]) -> dict[str, Any]:
    """Generates independent seed realizations (42, 43, 44) for the selected architecture."""
    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)
    cells = []

    arch = selected_model["architecture"]
    cmd_template = list(selected_model["command"])
    leads = selected_model.get("leads", [0, 1])

    for lead_idx in leads:
        lead_tag = f"l{lead_idx}"
        for seed in [42, 43, 44]:
            run_id = f"final_{arch}_s{seed}_{lead_tag}"
            run_dir = MULTISEED_DIR / "runs" / run_id

            cmd = []
            skip_next = False
            for token in cmd_template:
                if skip_next:
                    skip_next = False
                    continue
                if token == "--seed":
                    cmd.extend(["--seed", str(seed)])
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
                "seed": seed,
                "output_dir": str(run_dir),
                "command": cmd
            })

    manifest = {
        "version": 1,
        "experiment_name": "multiseed_tri_realization",
        "description": "Definitive 3-Seed Confirmation (theta_42, theta_43, theta_44) for Primary Architecture",
        "total_jobs": len(cells),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_architecture": arch,
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

def evaluate_stability_gate(results_dir: Path) -> dict[str, Any]:
    """Evaluates optimization stability gate and determines whether primary configuration passes or triggers fallback."""
    runs_dir = results_dir / "runs"
    if not runs_dir.is_dir():
        print("No multi-seed runs found.")
        return {}

    arch_seeds = {}
    for rdir in runs_dir.iterdir():
        sfile = rdir / "summary.json"
        if not sfile.is_file(): continue
        with open(sfile) as f: s = json.load(f)
        parts = rdir.name.split("_")
        lead = 0 if parts[-1] == "l0" else 1
        seed = int(parts[-2].replace("s", ""))
        arch = "_".join(parts[1:-2])

        key = (arch, lead)
        if key not in arch_seeds: arch_seeds[key] = {}
        arch_seeds[key][seed] = s

    print("\n" + "="*88)
    print("  PRESPECIFIED ENGINEERING / OPTIMIZATION STABILITY GATE (SEEDS 42, 43, 44)")
    print("="*88)

    gate_verdicts = {}
    for (arch, lead), seed_dict in arch_seeds.items():
        lead_tag = "Lead-I" if lead == 0 else "Lead-II"
        r_vals = [seed_dict[s]["val_missing_pearson"] for s in [42, 43, 44] if s in seed_dict]
        if len(r_vals) < 3:
            print(f"[{arch:<32} {lead_tag}] Incomplete seeds ({len(r_vals)}/3). Gate pending.")
            continue

        r_mean = float(np.mean(r_vals))
        r_std = float(np.std(r_vals))
        r_range = float(np.max(r_vals) - np.min(r_vals))

        passed = (r_range <= MAX_ALLOWED_SEED_RANGE) and (r_std <= MAX_ALLOWED_SEED_STD)
        gate_verdicts[(arch, lead)] = {
            "arch": arch,
            "lead": lead,
            "r_mean": r_mean,
            "r_std": r_std,
            "r_range": r_range,
            "passed": passed
        }

        status_str = "PASSED (Designating Primary Configuration)" if passed else "FAILED (Triggering Fallback List m_{k+1})"
        print(f"[{arch:<32} {lead_tag}] Mean r: {r_mean:.4f} | SD_seed: {r_std:.4f} | Range: {r_range:.4f} -> {status_str}")

    return gate_verdicts

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Tri-Realization Confirmation Pipeline (v3.3)")
    parser.add_argument("--auto-launch", action="store_true", help="Launch the queue immediately after generation")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze existing multi-seed results")
    args = parser.parse_args()

    if args.analyze_only:
        evaluate_stability_gate(MULTISEED_DIR)
        return

    print("Definitive Multi-Seed Confirmation Pipeline v3.3 Initialized.")

if __name__ == "__main__":
    main()
