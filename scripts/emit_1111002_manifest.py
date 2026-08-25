#!/usr/bin/env python3
"""Emit the 120-model wavelet SSL manifest for 1111002 (Lead 0 and Lead 1)."""

import json, sys, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_1lead_wavelet_ssl_mtl import broad_cells, cmd_for, parser, input_fingerprints

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            d.update(chunk)
    return d.hexdigest()

def main():
    base_manifest_path = ROOT / "refine-logs/wavelet_ssl_1110000/full/manifest.json"
    target_manifest_path = ROOT / "refine-logs/wavelet_ssl_1111002/full/manifest.json"
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    base = json.loads(base_manifest_path.read_text())
    
    # Construct args mock for cmd_for
    a = parser().parse_args([])
    a.delineation_dir = "data/rdb_wavelet_delineation_cache"
    a.data_dir = "data/ptb_xl/tensors"
    a.data_manifest = "refine-logs/ptbxl_tensor_content_manifest.json"
    a.sweep_output_root = "refine-logs/wavelet_ssl_1111002/full/runs"
    a.sweep_epochs = 3
    a.seed = 42
    a.batch_size = 32
    a.delineation_batch_size = 32
    a.num_workers = 4

    C = broad_cells(include_fiducials=True)
    print(f"Broad cells count: {len(C)}")
    
    jobs = []
    for lead in (0, 1):
        for c in C:
            n, cmd = cmd_for(c, a, lead, "1111002")
            jobs.append({
                "id": n,
                "status": "pending",
                "command": cmd,
                "cell": c
            })

    print(f"Total generated jobs: {len(jobs)} (Lead 0: {len(C)}, Lead 1: {len(C)})")
    
    inputs = input_fingerprints(a)
    
    asset_inventory = {}
    for job in jobs:
        c = job["cell"]
        for bank_key, asset_key in (("wavelet_bank", "custom_wavelet_asset"),
                                    ("view_a_bank", "view_a_custom_wavelet_asset"),
                                    ("view_b_bank", "view_b_custom_wavelet_asset")):
            if c.get(bank_key, "inherit") != "custom_asset":
                continue
            asset = Path(c[asset_key])
            if not asset.is_absolute():
                asset = ROOT / asset
            if not asset.is_file():
                raise FileNotFoundError(f"custom wavelet asset missing: {asset}")
            asset_inventory[str(asset.resolve())] = sha256_file(asset)

    payload = {
        "version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": len(C),
        "jobs": jobs,
        "checkpoint_retention": "rolling_resume_only; delete on success",
        **inputs,
        "custom_wavelet_assets": dict(sorted(asset_inventory.items())),
        "delineation_audit": base["delineation_audit"]
    }

    target_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Successfully emitted 1111002 manifest to: {target_manifest_path}")

if __name__ == "__main__":
    main()
