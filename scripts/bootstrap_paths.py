"""Shared import-path bootstrap for the benchmarking project."""

from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
ENGINEERING_ROOT = BENCHMARK_ROOT / "unified_latents" / "engineering"
ECG_FM_ROOT = BENCHMARK_ROOT / "ecg_fm_integration"
THIRD_PARTY_ROOT = ENGINEERING_ROOT / "third_party"


def setup_import_paths(*, include_fairseq: bool = False) -> None:
    """Add project roots to sys.path so local vendored packages resolve."""
    for p in (BENCHMARK_ROOT, ENGINEERING_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    if include_fairseq and ECG_FM_ROOT.exists():
        for sub in ("fairseq-signals", "fairseq"):
            p = ECG_FM_ROOT / sub
            if p.exists() and str(p) not in sys.path:
                sys.path.append(str(p))


def third_party_path(*parts: str) -> Path:
    return THIRD_PARTY_ROOT.joinpath(*parts)


def add_third_party_to_path(*parts: str) -> Path:
    """Append a local third_party subpath to sys.path if it exists."""
    p = third_party_path(*parts)
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))
    return p
