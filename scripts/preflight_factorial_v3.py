#!/usr/bin/env python3
"""Fail-closed preflight for the clinical-integrity repair queue."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scripts.echonext_classifier import (
    EchoNextMiniModel,
    SHD_LABEL_COLUMNS,
    load_echonext_test_metadata,
)
from scripts.evaluate_comprehensive_registry import LEAD_NAMES
from scripts.evaluate_smartwatch import DEVICES, GT_DEVICE, discover_records
from scripts.smartwatch_protocol import canonical_record_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiment_queue/factorial_v2/model_registry.json"),
    )
    parser.add_argument(
        "--comprehensive",
        type=Path,
        default=Path("results/factorial_v2/comprehensive_results.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_free_gib", type=float, default=3.0)
    args = parser.parse_args()
    registry_path = args.registry if args.registry.is_absolute() else PROJECT_ROOT / args.registry
    comprehensive_path = (
        args.comprehensive
        if args.comprehensive.is_absolute()
        else PROJECT_ROOT / args.comprehensive
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    checks: dict[str, object] = {}

    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    checks["free_disk_gib"] = free_bytes / 2**30
    require(
        free_bytes >= args.minimum_free_gib * 2**30,
        f"Need {args.minimum_free_gib:.1f} GiB free, found {free_bytes / 2**30:.2f}",
    )

    registry = json.loads(registry_path.read_text())
    models = registry["models"]
    ids = [model["id"] for model in models]
    require(len(models) == len(set(ids)) == 48, "Registry must contain 48 unique models")
    missing_checkpoints = [
        model["checkpoint"]
        for model in models
        if not (PROJECT_ROOT / model["checkpoint"]).is_file()
    ]
    require(not missing_checkpoints, f"Missing checkpoints: {missing_checkpoints}")
    checks["registry_models"] = len(models)

    comprehensive = json.loads(comprehensive_path.read_text())
    require(
        set(comprehensive["models"]) == set(ids),
        "PTB-XL result/registry IDs disagree",
    )
    for model in comprehensive["models"].values():
        for field in ("ecgfounder_per_record_parquet", "paper_parity_per_record_parquet"):
            require(
                (PROJECT_ROOT / model[field]).is_file(),
                f"Missing PTB-XL per-record artifact: {model.get(field)}",
            )
    checks["ptbxl_per_record_models"] = len(comprehensive["models"])
    checks["comprehensive_results"] = str(comprehensive_path)

    echo_dir = PROJECT_ROOT / "data/echonext"
    provenance = json.loads((echo_dir / "PROVENANCE.json").read_text())
    require(provenance["lead_order"] == LEAD_NAMES, "EchoNext lead order mismatch")
    require(provenance["units"] == "uV", "EchoNext pre-inversion units must be uV")
    require(
        provenance["normalization"]["kind"] == "dataset_zscore",
        "EchoNext official normalization provenance is required",
    )
    waveform_path = echo_dir / "EchoNext_test_waveforms.npy"
    waveform = np.load(waveform_path, mmap_mode="r")
    require(waveform.shape == (5442, 1, 2500, 12), f"EchoNext shape mismatch: {waveform.shape}")
    official_root = (
        PROJECT_ROOT
        / "ecg_fm_integration/echonext_minimodel_repo/7-EchoNext Minimodel"
    )
    shd = EchoNextMiniModel(official_root, device=__import__("torch").device("cpu"))
    metadata, tabular, labels = load_echonext_test_metadata(
        echo_dir / "echonext_metadata_100k.csv",
        shd.transformer_path,
    )
    require(
        labels.shape == (5442, len(SHD_LABEL_COLUMNS))
        and tabular.shape == (5442, 7)
        and len(metadata) == 5442,
        "EchoNext metadata/tabular/label alignment failed",
    )
    checks["echonext"] = {
        "waveform_shape": list(waveform.shape),
        "test_metadata_rows": len(metadata),
        "shd_tasks": labels.shape[1],
        "official_classifier": shd.provenance,
    }
    del shd

    watch_dir = (
        PROJECT_ROOT
        / "data/physionet.org/files/ecg-capable-smartwatches/1.0.0"
    )
    expected_counts = {
        "applewatch_serie8": 180,
        "fitbitsense2": 180,
        "samsunggalaxy6": 179,
        "withingsscanwatch": 180,
    }
    reference_records: dict[str, Path] = {}
    for reference_path in discover_records(watch_dir, GT_DEVICE):
        relative = reference_path.relative_to(watch_dir / GT_DEVICE)
        key = canonical_record_key(relative)
        require(key not in reference_records, f"Duplicate Philips record key: {key}")
        reference_records[key] = reference_path
    watch_checks: dict[str, object] = {}
    for device, expected in expected_counts.items():
        paired = 0
        lead_names: set[str] = set()
        for watch_path in discover_records(watch_dir, device):
            relative = watch_path.relative_to(watch_dir / device)
            reference_path = reference_records.get(canonical_record_key(relative))
            if reference_path is None:
                continue
            watch_header = wfdb.rdheader(str(watch_path))
            reference_header = wfdb.rdheader(str(reference_path))
            lead_names.update(watch_header.sig_name)
            require(
                reference_header.sig_name == LEAD_NAMES,
                f"Philips lead order mismatch: {reference_path}",
            )
            paired += 1
        require(paired == expected, f"{device}: expected {expected} pairs, found {paired}")
        require(lead_names == {"II"}, f"{device}: expected only lead II, found {lead_names}")
        watch_checks[device] = {"paired_records": paired, "input_leads": sorted(lead_names)}
    checks["smartwatch"] = watch_checks

    nstdb = PROJECT_ROOT / registry["nstdb_dir"]
    nstdb_records = {
        name: any(nstdb.glob(f"**/{name}.dat"))
        for name in ("bw", "em", "ma")
    }
    require(all(nstdb_records.values()), f"NSTDB records missing: {nstdb_records}")
    checks["nstdb"] = nstdb_records

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "minimum_free_gib": args.minimum_free_gib,
                "checks": checks,
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
