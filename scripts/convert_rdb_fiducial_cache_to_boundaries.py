#!/usr/bin/env python3
"""Convert the verified RDB 9-channel cache to its 6 released boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


OLD_NAMES = (
    "P_onset", "P_peak", "P_offset", "QRS_onset", "R_peak",
    "QRS_offset", "T_onset", "T_peak", "T_offset",
)
NEW_NAMES = (
    "P_onset", "P_offset", "QRS_onset", "QRS_offset", "T_onset", "T_offset",
)
KEEP = (0, 2, 3, 5, 6, 8)
DROP = (1, 4, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    manifest = json.loads((args.source / "manifest.json").read_text())
    args.output.mkdir(parents=True)
    totals = {"records": 0, "valid_boundaries": 0, "nonzero_boundaries": 0}
    for entry in manifest["records"]:
        relative = Path(entry["output"])
        source_path, output_path = args.source / relative, args.output / relative
        record = torch.load(source_path, map_location="cpu", weights_only=False)
        heatmaps = torch.as_tensor(record["fiducial_heatmaps"])
        valid = torch.as_tensor(record["fiducial_valid"], dtype=torch.bool)
        if tuple(record.get("fiducial_names", ())) != OLD_NAMES:
            raise ValueError(f"{source_path}: unexpected fiducial names")
        if heatmaps.shape != (12, 9, 5000) or valid.shape != (12, 9):
            raise ValueError(f"{source_path}: unexpected fiducial tensor shape")
        if valid[:, DROP].any() or heatmaps[:, DROP].count_nonzero():
            raise ValueError(f"{source_path}: removed peak channel contains supervision")
        record["fiducial_heatmaps"] = heatmaps[:, KEEP, :].contiguous()
        record["fiducial_valid"] = valid[:, KEEP].contiguous()
        record["fiducial_names"] = NEW_NAMES
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, output_path)
        entry["output_sha256"] = sha256(output_path)
        totals["records"] += 1
        totals["valid_boundaries"] += int(record["fiducial_valid"].sum())
        totals["nonzero_boundaries"] += int(record["fiducial_heatmaps"].count_nonzero())

    contract = manifest["label_contract"]
    contract["fiducial_names"] = list(NEW_NAMES)
    contract["peak_channel_policy"] = "no peak output channels; RDB releases only START/END"
    manifest["converted_at"] = datetime.now(timezone.utc).isoformat()
    manifest["conversion"] = {
        "source_cache": str(args.source.resolve()),
        "old_channels": list(OLD_NAMES),
        "selected_old_indices": list(KEEP),
        "asserted_empty_old_indices": list(DROP),
        **totals,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
