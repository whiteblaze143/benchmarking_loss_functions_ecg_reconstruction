"""Convert a canonical panorama into the unchanged repSpat data contract."""

from __future__ import annotations

import numpy as np


def panorama_to_anndata(panorama: np.ndarray, sampling_hz: float):
    """Create one temporal AnnData object with domain coordinates s_t=(t, 0)."""
    from anndata import AnnData

    values = np.asarray(panorama, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 8:
        raise ValueError("panorama must have shape [8, time]")
    if not np.isfinite(values).all():
        raise ValueError("panorama contains non-finite values")
    if not sampling_hz > 0:
        raise ValueError("sampling_hz must be positive")

    time_seconds = np.arange(values.shape[1], dtype=np.float64) / sampling_hz
    adata = AnnData(X=values.T)
    adata.var_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    adata.obsm["spatial"] = np.column_stack(
        [time_seconds, np.zeros_like(time_seconds)]
    )
    adata.layers["theta_panorama"] = adata.X.copy()
    adata.uns["theta_repspat"] = {
        "representation": "nefnet_v2_canonical_panorama",
        "input_leads": ["I", "II", "V3"],
        "sampling_hz": float(sampling_hz),
        "domain": "s_t=(t_seconds,0)",
    }
    return adata
