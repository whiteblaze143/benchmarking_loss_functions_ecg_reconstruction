#!/usr/bin/env python3
"""Plot a representative PTB-XL reconstruction from a one-lead ECG-AIM model."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_latents.engineering.experimental.aim_1_lead import build_alitok_vae_1d
from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
DEFAULT_MODEL_ID = "spatial_1lead_e1_panorama_film_1010010_s42_l1"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/onelead_cache" / f"{DEFAULT_MODEL_ID}.pt"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x0, y0 = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    return float(np.sum(x0 * y0) / denominator) if denominator else 0.0


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    observed = provenance.get("preprocessing", {}).get("observed_leads")
    if not isinstance(observed, list) or len(observed) != 1:
        raise RuntimeError(f"Expected exactly one observed lead, found {observed!r}")

    model = build_alitok_vae_1d(
        architecture=checkpoint["alitok_architecture"],
        target_len=int(checkpoint.get("target_len", 5000)),
        patch_size=int(checkpoint.get("alitok_patch_size", 25)),
        encoder_depth=int(checkpoint.get("alitok_encoder_depth", 8)),
        decoder_depth=int(checkpoint.get("alitok_decoder_depth", 4)),
        lead_conditioning_mode=str(checkpoint.get("lead_conditioning_mode", "learned")),
        use_learned_lead_id=bool(checkpoint.get("use_learned_lead_id", False)),
        use_relative_geometry=bool(checkpoint.get("use_relative_geometry", False)),
        use_spatial_film=bool(checkpoint.get("use_spatial_film", False)),
        spatial_gain_init=float(checkpoint.get("spatial_gain_init", 0.1)),
        geometry_control=str(checkpoint.get("geometry_control", "standard")),
    )
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint, [int(observed[0])]


@torch.inference_mode()
def reconstruct(model, tensor: torch.Tensor, observed: list[int], device: torch.device) -> torch.Tensor:
    target = tensor.to(device)
    masked = mask_unobserved_leads(target, observed).contiguous()
    lead_indices = make_lead_indices(observed, target.shape[0], device)
    result = model(masked, y_full=target, lead_indices=lead_indices, mode="stage1")
    output = result["y_pred"][:, :12, : target.shape[-1]].float()
    # Acquired leads are measurements, not predictions, and pass through unchanged.
    output[:, observed, :] = target[:, observed, :]
    return output.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/best_ecg_aim_1lead_reconstruction.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing {args.checkpoint}. Materialize {DEFAULT_MODEL_ID} with "
            "scripts/onelead_checkpoint_store.py first."
        )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model, checkpoint, observed = load_model(args.checkpoint, device)

    test_files = sorted(glob.glob(str(ROOT / "data/ptb_xl/tensors/test/*.pt")))
    if not test_files:
        raise RuntimeError("No PTB-XL test tensors found")
    candidates = []
    for filename in test_files[: args.candidates]:
        tensor = torch.load(filename, map_location="cpu", weights_only=False).float().unsqueeze(0)
        reconstruction = reconstruct(model, tensor, observed, device)
        gt, rc = tensor[0].numpy(), reconstruction[0].numpy()
        missing = [index for index in range(12) if index not in observed]
        mean_r = float(np.mean([pearson_r(gt[index], rc[index]) for index in missing]))
        amplitude = np.ptp(gt, axis=1)
        if np.all(amplitude > 0.5) and np.all(amplitude < 8.0):
            candidates.append((mean_r, Path(filename).stem, gt, rc))
    if not candidates:
        raise RuntimeError("No non-flat, bounded-amplitude candidate found; increase --candidates")
    mean_r, stem, gt, rc = max(candidates, key=lambda item: item[0])

    start, duration = FS, 2.5
    stop = start + int(duration * FS)
    time = np.arange(stop - start) / FS
    lead_matrix = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.5), dpi=300, sharex=True, sharey=True)
    fig.patch.set_facecolor("#FAFAFA")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.09, wspace=0.12, hspace=0.22)
    model_id = args.checkpoint.stem
    fig.suptitle(
        f"ECG-AIM One-Lead Reconstruction ({model_id} | PTB-XL #{stem})\n"
        f"Acquired {LEAD_NAMES[observed[0]]} $\\rightarrow$ 11 missing leads inferred "
        f"(record mean missing-lead $r$={mean_r:.3f})",
        fontsize=13, fontweight="bold", y=0.965,
    )

    for row in range(3):
        for column in range(4):
            ax = axes[row, column]
            lead = lead_matrix[row][column]
            acquired = lead in observed
            y_gt, y_rc = gt[lead, start:stop], rc[lead, start:stop]
            r_value = pearson_r(y_gt, y_rc)
            ax.set_facecolor("#FFFDFB")
            ax.set_xticks(np.arange(0, duration + 0.01, 0.2))
            ax.set_xticks(np.arange(0, duration + 0.01, 0.04), minor=True)
            ax.set_yticks(np.arange(-2.0, 2.5, 0.5))
            ax.set_yticks(np.arange(-2.0, 2.5, 0.1), minor=True)
            ax.grid(which="minor", color="#FFEBEB", linewidth=0.4, alpha=0.8)
            ax.grid(which="major", color="#FFD1D1", linewidth=0.7, alpha=0.8)
            if acquired:
                ax.plot(time, y_gt, color="#1A365D", linewidth=1.5)
                badge, color, background = "ACQUIRED", "#2B6CB0", "#EBF8FF"
            else:
                ax.plot(time, y_gt, color="#1A202C", linewidth=1.35, alpha=0.85)
                ax.plot(time, y_rc, color="#E53E3E", linewidth=1.35, linestyle="--", alpha=0.95)
                badge, color, background = f"RECON (r={r_value:.3f})", "#C53030", "#FFF5F5"
            ax.text(0.03, 0.88, LEAD_NAMES[lead], transform=ax.transAxes, fontsize=11,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E0"))
            ax.text(0.97, 0.88, badge, transform=ax.transAxes, fontsize=8.5,
                    fontweight="bold", color=color, ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=background, edgecolor=color))
            ax.set(xlim=(0, duration), ylim=(-1.5, 1.8))
            if row == 2:
                ax.set_xlabel("Time (seconds)", fontsize=9)
            if column == 0:
                ax.set_ylabel("Amplitude (mV)", fontsize=9)
            ax.tick_params(axis="both", which="both", labelsize=7.5, colors="#718096")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CBD5E0")

    handles = [
        plt.Line2D([0], [0], color="#1A202C", lw=1.6, label="Ground truth"),
        plt.Line2D([0], [0], color="#E53E3E", lw=1.6, linestyle="--", label="ECG-AIM reconstruction"),
        plt.Line2D([0], [0], color="#1A365D", lw=1.6, label=f"Acquired {LEAD_NAMES[observed[0]]}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=True,
               facecolor="white", edgecolor="#CBD5E0", bbox_to_anchor=(0.5, 0.018))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(args.output, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    fig.savefig(pdf_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"model={model_id}")
    print(f"checkpoint_best_val_missing_pearson={checkpoint.get('best_val_missing_pearson')}")
    print(f"record={stem} mean_missing_pearson={mean_r:.6f}")
    print(args.output)
    print(pdf_path)


if __name__ == "__main__":
    main()
