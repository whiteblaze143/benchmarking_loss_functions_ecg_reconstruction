#!/usr/bin/env python3
"""Emit the preflight manifest for wavelet SSL 1111002."""

import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import sha256_file, atomic_json

PREFLIGHT_CELLS = (
    "A0_raw",
    "E1_raw",
    "A0_wave_noSSL_gated_add",
    "A0_wave_noSSL_cross_attn",
    "ssl_magnitude_phase_sin_local_gated_add",
    "ssl_log_magnitude_phase_sin_both_cross_attn",
    "P1_A0_morlet_mag_phase_noSSL",
    "R0_morlet_mag_morlet_phase",
    "R1_morlet_mag_ueg_phase",
    "C1_E1_morlet_mag_morlet_phase",
    "ssl_global",
    "tf_sc48_cy6",
    "conv_control",
    "del_boundary",
    "del_fid",
)

def replace_option(command, flag, value):
    for i, part in enumerate(command):
        if part == flag and i + 1 < len(command):
            command[i + 1] = value
            return
    command.extend([flag, value])

def main():
    work_dir = ROOT / "refine-logs/wavelet_ssl_1111002"
    full_manifest_path = work_dir / "full/manifest.json"
    preflight_manifest_path = work_dir / "preflight/manifest.json"
    preflight_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    full = json.loads(full_manifest_path.read_text())
    full_sha = sha256_file(full_manifest_path)

    # Pick Lead 0 jobs for preflight
    selected = {}
    for job in full["jobs"]:
        cname = job["cell"]["name"]
        if cname in PREFLIGHT_CELLS and job["id"].endswith("_l0") and cname not in selected:
            selected[cname] = job

    missing = set(PREFLIGHT_CELLS) - set(selected)
    if missing:
        raise RuntimeError(f"Missing preflight cells in full manifest: {sorted(missing)}")

    jobs = []
    for ordinal, cell_name in enumerate(PREFLIGHT_CELLS):
        source = selected[cell_name]
        command = list(source["command"])
        run_name = f"preflight_{cell_name}_{ordinal:02d}"
        replace_option(command, "--run-name", run_name)
        replace_option(command, "--output-dir", str(work_dir / "preflight/runs" / run_name))
        command.append("--quick-verify")
        jobs.append({
            "id": run_name,
            "status": "pending",
            "command": command,
            "cell": source["cell"]
        })

    payload = {
        key: value for key, value in full.items()
        if key not in {"created_at", "cells", "jobs", "delineation_audit"}
    }
    payload.update({
        "version": 2,
        "created_at": full["created_at"],
        "cells": len(jobs),
        "jobs": jobs,
        "source_manifest_sha256": full_sha,
        "gate": "fifteen one-epoch/two-batch GPU branch-coverage configurations",
    })

    atomic_json(preflight_manifest_path, payload)
    print(f"Emitted {len(jobs)} preflight jobs to {preflight_manifest_path}")

if __name__ == "__main__":
    main()
