from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    "unet": "#0072B2",
    "multiscale_vae": "#E69F00",
    "ecg_aim": "#009E73",
    "reference": "#222222",
}
FAMILY_LABELS = {
    "unet": "U-Net",
    "multiscale_vae": "MultiScale-VAE",
    "ecg_aim": "ECG-AIM",
    "reference": "Reference",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


DEFAULT_SELECTION = Path("results/factorial_v2/selected_masks.json")


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_ids(selection_path: Path = DEFAULT_SELECTION) -> dict[str, str]:
    selection_path = resolve_project_path(selection_path)
    selected = json.loads(
        selection_path.read_text()
    )["masks"]
    prefixes = {
        "unet": "unet",
        "msvae": "multiscale_vae",
        "ecgaim": "ecg_aim",
    }
    ids = {}
    for family, mask in selected.items():
        if len(mask) == 3 and set(mask) <= {"0", "1"}:
            encoded = f"e1c{mask[0]}m{mask[1]}d{mask[2]}"
        elif len(mask) == 4 and set(mask) <= {"0", "1"}:
            if mask[0] != "1":
                raise ValueError(
                    f"Validation-selected mask must be MSE-on, found {family}={mask}"
                )
            encoded = f"e{mask[0]}c{mask[1]}m{mask[2]}d{mask[3]}"
        else:
            raise ValueError(f"Unsupported selected mask encoding: {family}={mask!r}")
        ids[prefixes[family]] = f"{family}__{encoded}__s42"
    return ids


def comparison_ids(selection_path: Path = DEFAULT_SELECTION) -> dict[str, dict[str, str]]:
    selected = selected_ids(selection_path)
    registry_prefix = {
        "unet": "unet",
        "multiscale_vae": "msvae",
        "ecg_aim": "ecgaim",
    }
    return {
        family: {
            "Base": f"{prefix}__e1c0m0d0__s42",
            "Validation-best": selected[family],
            "Full": f"{prefix}__e1c1m1d1__s42",
        }
        for family, prefix in registry_prefix.items()
    }


def save(
    fig: mpl.figure.Figure,
    output_base: Path,
    inputs: Iterable[Path],
    *,
    metadata: dict | None = None,
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in (".pdf", ".svg", ".png"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=300 if suffix == ".png" else None)
        outputs[suffix[1:]] = {"path": str(path), "sha256": sha256(path)}
    plt.close(fig)
    script_path = Path(sys.argv[0]).resolve()
    provenance = {
        "schema_version": 1,
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "common_helper_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            str(path): sha256(path)
            for path in inputs
        },
        "outputs": outputs,
        "metadata": metadata or {},
    }
    output_base.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False)
    )
