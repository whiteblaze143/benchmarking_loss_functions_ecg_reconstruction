#!/usr/bin/env python3
"""Emit the 48-job manifest and 8-job preflight manifest for Spatial Architecture Search V1."""

import json, sys, time, hashlib, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_1lead_spatial_arch import CELLS_48
from scripts.wavelet_ssl_queue import connect, initialize, sha256_file, atomic_json

PYTHON = "/home/mithunmanivannan/.venv/bin/python3"
PREFLIGHT_CELL_NAMES = [
    "A00_base_frozen",
    "A07_meta_film_leadwise",
    "A09_meta_hyperdecoder",
    "B15_target_query_selfattn",
    "D24_graph_patient",
    "F32_basis6_residual",
    "G35_source8_then_I",
    "H40_curriculum_8_4_2_1"
]

def main():
    work_dir = ROOT / "refine-logs/spatial_arch_1lead_v1"
    full_manifest_path = work_dir / "full/manifest.json"
    preflight_manifest_path = work_dir / "preflight/manifest.json"
    
    trainer_script = ROOT / "scripts/train_1lead_spatial_arch.py"
    model_script = ROOT / "unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py"
    data_manifest = ROOT / "refine-logs/ptbxl_tensor_content_manifest.json"
    
    full_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  EMITTING & INITIALIZING SPATIAL ARCHITECTURE SEARCH V1 MANIFESTS")
    print("="*70)

    # 1. Full 48-cell Lead I Manifest
    jobs_full = []
    for cell in CELLS_48:
        name = cell["name"]
        cmd = [
            PYTHON,
            str(trainer_script),
            "--run-name", name,
            "--output-dir", str(work_dir / "full/runs" / name),
            "--data-dir", "data/ptb_xl/tensors",
            "--data-manifest", "refine-logs/ptbxl_tensor_content_manifest.json",
            "--metadata-path", "refine-logs/spatial_arch_1lead_v1/assets/ptbxl_patient_metadata.parquet",
            "--cell-mode", cell["mode"],
            "--factorial-mask", "1110000",
            "--observed-lead", "0",
            "--epochs", "3",
            "--batch-size", "32",
            "--num-workers", "4",
            "--seed", "42"
        ]
        if cell.get("shuffle_meta"):
            cmd.append("--shuffle-metadata")
        if cell.get("random_source"):
            cmd.append("--random-source")
        if cell.get("width"):
            cmd.extend(["--width", str(cell["width"])])
            
        jobs_full.append({
            "id": name,
            "status": "pending",
            "command": cmd,
            "cell": cell
        })

    payload_full = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "program": "patient_spatial_arch_v1",
        "cells": len(jobs_full),
        "jobs": jobs_full,
        "observed_lead": 0,
        "reconstruction_loss": "1110000",
        "metadata_asset": "refine-logs/spatial_arch_1lead_v1/assets/ptbxl_patient_metadata.parquet",
        "trainer_sha256": sha256_file(trainer_script),
        "model_sha256": sha256_file(model_script),
        "data_manifest_sha256": sha256_file(data_manifest),
    }

    atomic_json(full_manifest_path, payload_full)
    print(f"1. Emitted full manifest ({len(jobs_full)} jobs) -> {full_manifest_path}")

    # 2. 8-cell Preflight Manifest
    jobs_preflight = []
    for name in PREFLIGHT_CELL_NAMES:
        source_job = next(j for j in jobs_full if j["id"] == name)
        cmd_pf = list(source_job["command"])
        pf_name = f"preflight_{name}"
        for i in range(len(cmd_pf)):
            if cmd_pf[i] == "--run-name": cmd_pf[i+1] = pf_name
            if cmd_pf[i] == "--output-dir": cmd_pf[i+1] = str(work_dir / "preflight/runs" / pf_name)
        cmd_pf.append("--quick-verify")
        jobs_preflight.append({
            "id": pf_name,
            "status": "pending",
            "command": cmd_pf,
            "cell": source_job["cell"]
        })

    payload_preflight = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "program": "patient_spatial_arch_v1_preflight",
        "cells": len(jobs_preflight),
        "jobs": jobs_preflight,
        "gate": "8 representative architecture branches",
        "trainer_sha256": sha256_file(trainer_script),
        "model_sha256": sha256_file(model_script),
        "data_manifest_sha256": sha256_file(data_manifest),
    }

    atomic_json(preflight_manifest_path, payload_preflight)
    print(f"2. Emitted preflight manifest ({len(jobs_preflight)} jobs) -> {preflight_manifest_path}")

    # 3. Initialize SQLite databases
    print("\n3. Initializing SQLite Databases...")
    preflight_db = work_dir / "preflight/queue.sqlite"
    full_db = work_dir / "full/queue.sqlite"

    with connect(preflight_db) as conn:
        initialize(conn, preflight_manifest_path, ROOT)
    with connect(full_db) as conn:
        initialize(conn, full_manifest_path, ROOT)

    print(f"   Preflight DB ({preflight_db.name}) verified.")
    print(f"   Full DB ({full_db.name}) verified.")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
