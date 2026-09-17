from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_superdiagnostic_map(path: str | Path) -> dict[str, str]:
    statements = pd.read_csv(path, index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1]
    mapping = diagnostic["diagnostic_class"].dropna().astype(str).to_dict()
    return mapping


def parse_scp_codes(value: str) -> dict[str, float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must parse to a dictionary")
    return {str(k): float(v) for k, v in parsed.items()}


def encode_superdiagnostic(
    value: str,
    mapping: dict[str, str],
    classes: tuple[str, ...],
) -> np.ndarray:
    codes = parse_scp_codes(value)
    present = {mapping[code] for code in codes if code in mapping}
    return np.asarray([name in present for name in classes], dtype=np.float32)
