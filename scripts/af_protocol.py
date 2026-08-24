"""Shared, frozen contracts for the AF-preservation evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_MAP = ROOT / "configs/af_label_map.yaml"
RDB_GROUPS = ("AF", "AFL", "AFIB", "SI", "SA", "AT", "SVT", "VT", "SVT/VT", "ST", "SR", "SB")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONDITIONS = ("real12", "source", "real11", "recon12", "synthetic11")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_label_map(path: Path = DEFAULT_LABEL_MAP) -> dict[str, Any]:
    mapping = yaml.safe_load(path.read_text())
    if mapping.get("version") != 1 or mapping.get("endpoint") != "strict_atrial_fibrillation":
        raise ValueError(f"unsupported AF label map: {path}")
    if mapping["ptbxl"]["positive"] != ["AFIB"] or mapping["ptbxl"]["atrial_flutter"] != ["AFLT"]:
        raise ValueError("PTB-XL AF/AFL contract changed")
    if mapping["rdb"]["positive"] != ["AF"] or set(mapping["rdb"]["atrial_flutter"]) != {"AFL", "AFIB"}:
        raise ValueError("RDB AF/AFL contract changed")
    return mapping


def parse_scp_codes(value: str) -> set[str]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must decode to a dictionary")
    # The frozen strict-AF endpoint is statement-key presence, exactly as
    # prespecified. PTB-XL legitimately contains rhythm statements with 0.0
    # likelihood; filtering on the numeric value would discard most AFIB rows.
    return {str(code) for code in parsed}


def ptbxl_af_label(codes: set[str]) -> tuple[int | None, str]:
    af, flutter = "AFIB" in codes, "AFLT" in codes
    if af and flutter:
        return None, "conflict_AFIB_AFLT"
    return (1, "AF") if af else (0, "AFL" if flutter else "other")


def rdb_af_label(code: str) -> tuple[int, str]:
    normalized = str(code).strip().upper()
    if normalized == "AF":
        return 1, "AF"
    if normalized in {"AFL", "AFIB"}:
        return 0, "AFL"
    if normalized in {"SI", "SA"}:
        return 0, "SI"
    if normalized in {"SVT", "VT", "SVT/VT"}:
        return 0, "SVT/VT"
    if normalized not in set(RDB_GROUPS):
        raise ValueError(f"unrecognized RDB rhythm: {code!r}")
    return 0, normalized


def lead_mask(condition: str, source_index: int) -> list[int]:
    if source_index not in (0, 1):
        raise ValueError("AF protocol supports source lead I or II only")
    if condition == "real12" or condition == "recon12":
        return list(range(12))
    if condition == "source":
        return [source_index]
    if condition == "real11" or condition == "synthetic11":
        return [index for index in range(12) if index != source_index]
    raise ValueError(f"unknown AF condition: {condition}")
