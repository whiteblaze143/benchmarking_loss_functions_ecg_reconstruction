"""Shared utilities for ENTRENCH-AF pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
CONFIG_DIR = EXP_ROOT / "config"
RESULTS_ROOT = EXP_ROOT / "results"
DEFINITIONS_PATH = CONFIG_DIR / "definitions.yaml"
COHORTS_PATH = CONFIG_DIR / "cohorts.yaml"
LATEST_RUN_POINTER = RESULTS_ROOT / "LATEST_RUN.txt"

COHORT_MAP = {
    "IRIDIA-AF": "IRIDIA",
    "IRIDIA": "IRIDIA",
    "SHDB-AF": "SHDB",
    "SHDB": "SHDB",
    "LTAFDB": "LTAFDB",
}


def load_definitions() -> dict[str, Any]:
    with open(DEFINITIONS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cohorts() -> dict[str, Any]:
    with open(COHORTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_definitions_hash() -> str:
    return sha256_file(DEFINITIONS_PATH)


def get_git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def utc_timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_timestamp_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_run_dir() -> tuple[Path, str, str]:
    """Create a fresh run directory results/run_<timestamp>_<githash>/ and update LATEST_RUN.txt."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ts_compact = utc_timestamp_compact()
    ts_iso = utc_timestamp_iso()
    git_hash = get_git_hash()
    run_dir = RESULTS_ROOT / f"run_{ts_compact}_{git_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_POINTER.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    return run_dir, git_hash, ts_iso


def resolve_run_dir(run_dir: str | Path | None = None, create_if_missing: bool = False) -> Path:
    """Resolve run directory from explicit arg or LATEST_RUN.txt."""
    if run_dir is not None:
        p = Path(run_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    if LATEST_RUN_POINTER.exists():
        txt = LATEST_RUN_POINTER.read_text(encoding="utf-8").strip()
        if txt:
            p = Path(txt)
            if p.exists():
                return p
    if create_if_missing:
        p, _, _ = create_run_dir()
        return p
    raise FileNotFoundError(
        f"No run directory found. Run Stage 0 first or pass --run-dir. Expected pointer: {LATEST_RUN_POINTER}"
    )


def canonical_cohort(name: str) -> str:
    if name not in COHORT_MAP:
        raise ValueError(f"Unknown cohort name: {name!r}")
    return COHORT_MAP[name]


def stamp_dataframe(
    df: pd.DataFrame,
    git_hash: str,
    definitions_hash: str,
    run_timestamp: str,
) -> pd.DataFrame:
    out = df.copy()
    out["git_hash"] = git_hash
    out["definitions_hash"] = definitions_hash
    out["run_timestamp"] = run_timestamp
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
