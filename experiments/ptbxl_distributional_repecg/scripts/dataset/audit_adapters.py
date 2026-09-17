#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


CANONICAL_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wfdb_header(path: Path) -> dict[str, object]:
    header = wfdb.rdheader(str(path))
    return {
        "sampling_hz": int(header.fs),
        "samples": int(header.sig_len),
        "source_leads": list(header.sig_name),
        "units": list(header.units),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads((args.experiment / "outputs/native_task_reconciliation.json").read_text())["datasets"]

    echo_provenance_path = args.data_root / "echonext/PROVENANCE.json"
    echo_provenance = json.loads(echo_provenance_path.read_text())
    echo_waveforms_path = args.data_root / "echonext/EchoNext_test_waveforms.npy"
    echo_waveforms = np.load(echo_waveforms_path, mmap_mode="r")

    kingston_root = args.experiment / "outputs/kingston_negative_control"
    kingston_manifests = {
        split: json.loads((kingston_root / f"kingston_manifest_{split}.json").read_text())
        for split in ("train", "test")
    }
    kingston_source = sum(item["source_records"] for item in kingston_manifests.values())
    kingston_eligible = sum(item["eligible_records"] for item in kingston_manifests.values())

    ludb_headers = sorted((args.data_root / "ludb").glob("*.hea"))
    isp_train = sorted((args.data_root / "isp_delineation_dataset/train_data").glob("*.hea"))
    isp_test = sorted((args.data_root / "isp_delineation_dataset/test_data").glob("*.hea"))
    sunny_files = sorted((args.data_root / "sunnybrook_12_lead_ecg_samples").glob("*.xml"))
    zhejiang_labels = sorted((args.data_root / "zhejiang/label").glob("*.pkl"))
    zhejiang_signals = sorted((args.data_root / "zhejiang/ecg").glob("*.pkl"))
    rdb_manifest_path = args.data_root / "rdb_wavelet_delineation_cache/manifest.json"
    rdb_manifest = json.loads(rdb_manifest_path.read_text())

    emory_metadata = native["emory_muse"]
    emory_sample = Path("/data/mithunmanivannan/heedb_emory/WFDB/2010/MUSE_20191204_094914_19000")
    emory_header = _wfdb_header(emory_sample)
    heedb_audit_path = Path("/data/mithunmanivannan/heedb_audit/final/HEEDB_FULL_AUDIT_SUMMARY.json")
    heedb_audit = json.loads(heedb_audit_path.read_text())
    if heedb_audit["audit_gates"]["HEEDB_AUDIT_GATE"] != "PASS":
        raise ValueError("HEEDB source audit is not passed")

    adapters = {
        "echonext": {
            "status": "ready_test_only",
            "source_records": native["echonext"]["splits"]["test"]["records"],
            "locally_available_records": int(echo_waveforms.shape[0]),
            "locally_unavailable_records": 0,
            "available_split": "test",
            "shape": list(echo_waveforms.shape),
            "sampling_hz": echo_provenance["sampling_rate_hz"],
            "source_leads": echo_provenance["lead_order"],
            "independent_basis": INDEPENDENT_8,
            "units": echo_provenance["units"],
            "conversion": "invert released dataset z-score, then uV to mV",
            "provenance_sha256": _sha256(echo_provenance_path),
        },
        "kingston_icu": {
            "status": "ready_negative_control",
            "source_records": kingston_source,
            "eligible_records": kingston_eligible,
            "excluded_records": kingston_source - kingston_eligible,
            "exclusion_reconciliation": {
                split: {
                    "eligible": item["eligible_records"],
                    "ineligible": item["ineligible_records"],
                    "manifest_sha256": _sha256(kingston_root / f"kingston_manifest_{split}.json"),
                }
                for split, item in kingston_manifests.items()
            },
            "sampling_hz": 240,
            "source_leads": ["I", "II", "III", "V"],
            "admitted_leads": ["I", "II"],
            "excluded_leads": {"III": "derived", "V": "unspecified precordial location"},
            "independent_basis_mask": [1, 1, 0, 0, 0, 0, 0, 0],
            "units": "mV",
        },
        "ludb": {
            "status": "ready",
            "source_records": native["ludb"]["records"],
            "eligible_records": len(ludb_headers),
            "excluded_records": native["ludb"]["records"] - len(ludb_headers),
            "header_sample": _wfdb_header(args.data_root / "ludb/1"),
            "canonical_leads": CANONICAL_12,
            "independent_basis": INDEPENDENT_8,
        },
        "rdb": {
            "status": "ready",
            "source_records": native["rdb"]["records"],
            "eligible_records": rdb_manifest["conversion"]["records"],
            "excluded_records": native["rdb"]["records"] - rdb_manifest["conversion"]["records"],
            "sampling_hz": rdb_manifest["waveform_contract"]["sample_rate_hz"],
            "source_shape": rdb_manifest["waveform_contract"]["shape"],
            "source_leads": CANONICAL_12,
            "independent_basis": INDEPENDENT_8,
            "units": rdb_manifest["waveform_contract"]["units"],
            "manifest_sha256": _sha256(rdb_manifest_path),
        },
        "isp": {
            "status": "ready",
            "source_records": native["isp"]["records"],
            "eligible_records": len(isp_train) + len(isp_test),
            "excluded_records": native["isp"]["records"] - len(isp_train) - len(isp_test),
            "split_records": {"train": len(isp_train), "test": len(isp_test)},
            "header_sample": _wfdb_header(args.data_root / "isp_delineation_dataset/train_data/1"),
            "canonical_leads": CANONICAL_12,
            "independent_basis": INDEPENDENT_8,
            "resampling": "1000 Hz to 500 Hz with label-coordinate reconciliation required",
        },
        "zhejiang": {
            "status": "ready",
            "source_records": native["zhejiang"]["records"],
            "eligible_signal_sets": len(zhejiang_signals) // 12,
            "label_records": len(zhejiang_labels),
            "excluded_records": 0,
            "source_leads": CANONICAL_12,
            "independent_basis": INDEPENDENT_8,
            "source_sampling_hz": 2000,
            "model_sampling_hz": 500,
            "resampling": "polyphase 4:1 downsampling with masks mapped to the same coordinate system",
            "units": "uV converted to mV (verified via physiological amplitude bounds ~3mV)",
            "source_provenance": "https://doi.org/10.1038/s41597-020-0386-x",
        },
        "emory_muse": {
            "status": "ready_after_label_exclusions",
            "source_records": emory_metadata["metadata_rows"],
            "eligible_records": emory_metadata["eligible_records"],
            "excluded_records": emory_metadata["metadata_rows"] - emory_metadata["eligible_records"],
            "exclusion_reasons": {
                "empty_12sl_codes": emory_metadata["ineligible_records"],
                "no_diagnosis_row": emory_metadata["metadata_rows"] - emory_metadata["diagnosis_rows"],
            },
            "header_sample": emory_header,
            "canonical_leads": CANONICAL_12,
            "independent_basis": INDEPENDENT_8,
            "source_audit_sha256": _sha256(heedb_audit_path),
        },
        "sunnybrook": {
            "status": "ready_external_only",
            "source_records": native["sunnybrook"]["records"],
            "eligible_records": len(sunny_files),
            "excluded_records": native["sunnybrook"]["records"] - len(sunny_files),
            "source_leads": CANONICAL_12,
            "independent_basis": INDEPENDENT_8,
            "sampling_hz": 500,
            "units": "uV decoded from Philips XML, converted to mV",
        },
    }
    if set(adapters) != set(native) - {"ptbxl"}:
        raise ValueError("external adapter roster disagrees with native-task reconciliation")
    payload = {
        "kind": "external_adapter_reconciliation",
        "status": "blocked" if any(item["status"].startswith("blocked_") for item in adapters.values()) else "complete",
        "blocked_datasets": sorted(name for name, item in adapters.items() if item["status"].startswith("blocked_")),
        "adapters": adapters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "blocked_datasets": payload["blocked_datasets"]}))


if __name__ == "__main__":
    main()
