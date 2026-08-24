"""Shared poster plotting style for the isolated v4 candidate figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FAMILY_ORDER = ["unet", "msvae", "ecgaim"]
FAMILY_LABELS = {
    "unet": "U-Net",
    "msvae": "MultiScale-VAE",
    "ecgaim": "ECG-AIM",
}
FAMILY_COLORS = {
    "unet": "#0072B2",
    "msvae": "#009E73",
    "ecgaim": "#D55E00",
}
COMPONENT_COLORS = {
    "mse": "#000000",
    "correlation": "#0072B2",
    "mmd": "#E69F00",
    "derivative": "#CC79A7",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )


def save_triplet(
    fig: plt.Figure,
    output_base: Path,
    *,
    sources: list[Path],
    caption: str,
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "svg", "png"):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path)
        outputs[suffix] = {
            "path": str(path),
            "sha256": sha256(path),
        }
    provenance = {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()),
        "sources": {str(path): sha256(path) for path in sources},
        "outputs": outputs,
        "caption": caption,
        "style": "poster_v2_colorblind_vector",
    }
    output_base.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    plt.close(fig)
