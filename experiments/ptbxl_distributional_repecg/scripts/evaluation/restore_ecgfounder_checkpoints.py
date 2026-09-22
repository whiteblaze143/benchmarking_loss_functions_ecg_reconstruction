#!/usr/bin/env python3
"""Restore and strictly validate the two archived ECGFounder checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import torch


ARCHIVE_MEMBERS = {
    "1_lead": "home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/ecg_fm_integration/ecgfounder_repo/checkpoint/1_lead_ECGFounder.pth",
    "12_lead": "home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/ecg_fm_integration/ecgfounder_repo/checkpoint/12_lead_ECGFounder.pth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(repo: Path, leads: int) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    from net1d import Net1D
    return Net1D(
        in_channels=leads, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16,
        stride=2, groups_width=16, n_classes=150, use_bn=False,
        use_do=False, return_features=True, verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {args.output}")
    if not args.archive.is_file() or not args.archive_sha256.is_file():
        raise FileNotFoundError("archive or archive checksum file is missing")
    expected = args.archive_sha256.read_text().strip().split()[0]
    actual = sha256(args.archive)
    if actual != expected:
        raise RuntimeError(f"archive SHA-256 mismatch: expected {expected}, got {actual}")

    args.output.mkdir(parents=True)
    restored: dict[str, dict[str, object]] = {}
    with tarfile.open(args.archive, "r:gz") as bundle:
        names = {member.name for member in bundle.getmembers()}
        for name, member_name in ARCHIVE_MEMBERS.items():
            if member_name not in names:
                raise RuntimeError(f"required archive member missing: {member_name}")
            destination = args.output / f"ecgfounder_{name}.pth"
            source = bundle.extractfile(member_name)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member_name}")
            with destination.open("xb") as handle:
                handle.write(source.read())
            checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise RuntimeError(f"{destination} does not contain checkpoint['state_dict']")
            model = build_model(args.repo, 1 if name == "1_lead" else 12)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            restored[name] = {
                "archive_member": member_name,
                "path": str(destination),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "strict_state_load": True,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }

    payload = {
        "archive": str(args.archive), "archive_sha256": actual,
        "source_repo": str(args.repo), "restored": restored,
    }
    (args.output / "admission.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
