#!/usr/bin/env python3
"""Generate evidence-locked visual assets for the EMBC 2026 LaTeX poster."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "final_poster"
FIG = OUT / "assets" / "figures"
DATA = OUT / "data"
LOGOS = OUT / "assets" / "logos"
TEMPLATE = OUT / "assets" / "template"
QR = OUT / "assets" / "qr"

MASTER_PATH = (
    ROOT
    / "results"
    / "comprehensive_latest_48_models"
    / "all_48_models_master.csv"
)
FAMILYWISE_PATH = ROOT / "results" / "factorial_v4_2x4" / "familywise_endpoint_tests.csv"
EFFECTS_PATH = (
    ROOT / "results" / "factorial_v4_2x4" / "factorial_effects_mse_on_conditional.csv"
)

BLUE = "#00629B"
DARK = "#17324D"
TEAL = "#00A6A6"
GOLD = "#F2B134"
LIGHT_BLUE = "#E9F4FA"
LIGHT_GOLD = "#FFF4D6"
MUTED = "#536273"
GRID = "#D8E1E8"
FAMILY_COLORS = {"unet": BLUE, "multiscale_vae": GOLD, "ecg_aim": TEAL}
FAMILY_LABELS = {
    "unet": "U-Net",
    "multiscale_vae": "MultiScale-VAE",
    "ecg_aim": "ECG-AIM",
}
STATS_LABELS = {"unet": "U-Net", "msvae": "MultiScale-VAE", "ecgaim": "ECG-AIM"}
MASKS = [f"{i:04b}" for i in range(16)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup() -> None:
    for path in (FIG, DATA, LOGOS, TEMPLATE, QR):
        path.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": DARK,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def copy_locked_inputs() -> None:
    sources = {
        "all_48_models_master.csv": MASTER_PATH,
        "familywise_endpoint_tests.csv": FAMILYWISE_PATH,
        "factorial_effects_mse_on_conditional.csv": EFFECTS_PATH,
        "completeness_verification.json": (
            ROOT / "results" / "comprehensive_latest_48_models" / "VERIFICATION.json"
        ),
    }
    for name, source in sources.items():
        shutil.copy2(source, DATA / name)

    official = ROOT / "poster_final" / "template_media"
    for name in ("image1.png", "image2.png", "image3.png", "image4.png", "image5.png"):
        shutil.copy2(official / name, TEMPLATE / name)
    shutil.copy2(official / "image6.png", LOGOS / "ieee_embs.png")
    shutil.copy2(official / "image7.png", LOGOS / "embc2026.png")
    shutil.copy2(ROOT / "poster_html" / "assets" / "qr" / "project_qr.png", QR / "project_qr.png")
    shutil.copy2(
        ROOT / "IEEE-EMBSPosterTemplate2026-1 (1).pptx",
        TEMPLATE / "IEEE-EMBSPosterTemplate2026.pptx",
    )


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == "png":
            kwargs["dpi"] = 320
        fig.savefig(FIG / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def read_master() -> pd.DataFrame:
    frame = pd.read_csv(MASTER_PATH)
    frame["mask"] = frame["factorial_mask"].map(lambda value: f"{int(value):04d}")
    return frame


def make_factorial_atlas(master: pd.DataFrame) -> None:
    metrics = [
        ("ptbxl_missing_r2", r"Missing-lead $R^2$", (-0.2, 0.85), 2),
        ("ptbxl_qrs_correlation", "QRS correlation", (0.0, 0.95), 2),
        ("ptbxl_st_correlation", "ST correlation", (0.0, 0.90), 2),
        ("ecgfounder_150_macro_auroc", "ECGFounder macro AUROC", (0.68, 0.89), 3),
    ]
    families = ["unet", "multiscale_vae", "ecg_aim"]
    cmap = LinearSegmentedColormap.from_list(
        "embc_blue", ["#EAF4FA", "#86C5E4", BLUE, DARK]
    )
    fig, axes = plt.subplots(4, 1, figsize=(13.2, 7.9), constrained_layout=True)

    for ax, (column, title, limits, decimals) in zip(axes, metrics):
        matrix = np.zeros((3, 16))
        for row_index, family in enumerate(families):
            subset = master[master.family.eq(family)].set_index("mask")
            matrix[row_index] = [subset.loc[mask, column] for mask in MASKS]
        shown = np.clip(matrix, *limits)
        image = ax.imshow(shown, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
        ax.set_title(title, loc="left", color=DARK, pad=5)
        ax.set_yticks(range(3), [FAMILY_LABELS[value] for value in families])
        ax.set_xticks(range(16), MASKS)
        ax.tick_params(axis="x", rotation=0)
        for i in range(3):
            for j in range(16):
                value = matrix[i, j]
                text = (
                    f"<{limits[0]:.1f}"
                    if value < limits[0]
                    else f"{value:.{decimals}f}"
                )
                color = "white" if shown[i, j] > (limits[0] + 0.62 * (limits[1] - limits[0])) else DARK
                ax.text(j, i, text, ha="center", va="center", fontsize=7.2, color=color)
        # MSE-only (1000) and full (1111) are the prespecified comparison.
        for column_index, edge, linewidth in ((8, GOLD, 2.4), (15, TEAL, 2.4)):
            for row_index in range(3):
                ax.add_patch(
                    plt.Rectangle(
                        (column_index - 0.49, row_index - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor=edge,
                        linewidth=linewidth,
                    )
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.013, pad=0.008)
        cbar.ax.tick_params(labelsize=7)
    axes[-1].set_xlabel(
        "Loss mask E/C/M/D (gold outline = MSE-only 1000; teal outline = full 1111)",
        color=DARK,
        labelpad=7,
    )
    fig.suptitle(
        r"Complete $2^4$ factorial atlas: 48 independently trained cells",
        color=DARK,
        fontsize=16,
        fontweight="bold",
    )
    save(fig, "factorial_atlas")


def make_clinical_effects() -> None:
    frame = pd.read_csv(FAMILYWISE_PATH)
    morphology = frame[frame.endpoint.isin(["QRS", "ST"])].copy()
    diagnostic = frame[frame.endpoint.eq("diagnostic_utility")].copy()
    order = ["unet", "msvae", "ecgaim"]
    colors = {"unet": BLUE, "msvae": GOLD, "ecgaim": TEAL}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.5),
        gridspec_kw={"width_ratios": [1.45, 1]},
        constrained_layout=True,
    )
    ax = axes[0]
    y = 0
    yticks, ylabels = [], []
    for family in order:
        for endpoint, marker in (("QRS", "o"), ("ST", "s")):
            row = morphology[(morphology.family.eq(family)) & (morphology.endpoint.eq(endpoint))].iloc[0]
            estimate, low, high = row[["estimate", "ci_low", "ci_high"]]
            ax.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                ms=8,
                capsize=4,
                lw=2,
                color=colors[family],
            )
            yticks.append(y)
            ylabels.append(f"{STATS_LABELS[family]} · {endpoint}")
            y += 1
        y += 0.45
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(yticks, ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Full composite − MSE-only correlation")
    ax.set_title("Morphology improves in all architectures", color=DARK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.text(
        0.99,
        0.02,
        r"All six: $p<0.0167$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=DARK,
        bbox={"boxstyle": "round,pad=.25", "facecolor": LIGHT_GOLD, "edgecolor": GOLD},
    )

    ax = axes[1]
    for i, family in enumerate(order):
        row = diagnostic[diagnostic.family.eq(family)].iloc[0]
        estimate, low, high = row[["estimate", "ci_low", "ci_high"]]
        ax.errorbar(
            estimate,
            i,
            xerr=[[estimate - low], [high - estimate]],
            fmt="D",
            ms=8,
            capsize=4,
            lw=2,
            color=colors[family],
        )
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(3), [STATS_LABELS[value] for value in order])
    ax.invert_yaxis()
    ax.set_xlabel("Full composite − MSE-only AUROC")
    ax.set_title("ECGFounder AUROC: no significant change", color=DARK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.suptitle(
        "Prespecified patient-cluster BCa tests (2,000 resamples; 1,904 patients)",
        fontsize=15,
        color=DARK,
        fontweight="bold",
    )
    save(fig, "clinical_effects")


def make_noise_small_multiples(master: pd.DataFrame) -> None:
    full = master[master["mask"].eq("1111")].set_index("family")
    families = ["unet", "multiscale_vae", "ecg_aim"]
    snr = np.array([24, 12, 6, 0])
    panels = [
        ("Gaussian", "gaussian"),
        ("Baseline wander", "nstdb_bw"),
        ("Electrode motion", "nstdb_em"),
        ("Muscle artifact", "nstdb_ma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), sharey=True, constrained_layout=True)
    for ax, (title, prefix) in zip(axes, panels):
        for family in families:
            values = [
                full.loc[family, f"noise_{prefix}_{level}db_missing_mse"]
                for level in snr
            ]
            ax.plot(
                snr,
                values,
                "-o",
                color=FAMILY_COLORS[family],
                lw=2,
                ms=5,
                label=FAMILY_LABELS[family],
            )
        ax.invert_xaxis()
        ax.set_xticks(snr)
        ax.set_title(title, color=DARK)
        ax.set_xlabel("SNR (dB)")
        ax.grid(color=GRID, lw=0.7)
    axes[0].set_ylabel("Missing-lead MSE (mV²)")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Deterministically paired noise stress test — full composite loss",
        fontsize=15,
        color=DARK,
        fontweight="bold",
    )
    save(fig, "noise_stress")


def make_external_scorecard(master: pd.DataFrame) -> None:
    full = master[master["mask"].eq("1111")].set_index("family")
    families = ["unet", "multiscale_vae", "ecg_aim"]
    labels = [FAMILY_LABELS[value] for value in families]
    colors = [FAMILY_COLORS[value] for value in families]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.15),
        gridspec_kw={"width_ratios": [1.1, 1.35, 1.0]},
        constrained_layout=True,
    )
    ax = axes[0]
    datasets = [
        ("PTB-XL", "ptbxl_missing_pearson"),
        ("EchoNext", "echonext_missing_pearson"),
        ("Sunnybrook", "sunnybrook_missing_pearson"),
    ]
    x = np.arange(3)
    width = 0.23
    for i, family in enumerate(families):
        values = [full.loc[family, column] for _, column in datasets]
        ax.bar(x + (i - 1) * width, values, width, color=colors[i], label=labels[i])
    ax.set_xticks(x, [value[0] for value in datasets])
    ax.set_ylim(0.70, 0.96)
    ax.set_ylabel("Missing-lead Pearson")
    ax.set_title("Waveform transfer", color=DARK)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    devices = [
        ("Apple", "applewatch_serie8"),
        ("Fitbit", "fitbitsense2"),
        ("Samsung", "samsunggalaxy6"),
        ("Withings", "withingsscanwatch"),
    ]
    matrix = np.array(
        [
            [
                full.loc[family, f"smartwatch_{device}_missing11_pearson"]
                for _, device in devices
            ]
            for family in families
        ]
    )
    image = ax.imshow(matrix, cmap="Blues", vmin=0.2, vmax=0.7, aspect="auto")
    ax.set_xticks(range(4), [value[0] for value in devices])
    ax.set_yticks(range(3), labels)
    for i in range(3):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if matrix[i, j] > 0.50 else DARK,
            )
    ax.set_title("Smartwatch Lead I → 11 targets", color=DARK)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)

    ax = axes[2]
    endpoints = [
        ("ECGFounder\n150-task", "ecgfounder_150_macro_auroc"),
        ("Signal quality\nmacro", "ptbxl_signal_quality_macro_auroc"),
        ("Sunnybrook\nproxy", "sunnybrook_proxy_superclass_auroc"),
    ]
    x = np.arange(3)
    for i, family in enumerate(families):
        values = [full.loc[family, column] for _, column in endpoints]
        ax.scatter(
            x + (i - 1) * 0.12,
            values,
            s=70,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.set_xticks(x, [value[0] for value in endpoints])
    ax.set_ylim(0.56, 0.91)
    ax.set_ylabel("Macro AUROC")
    ax.set_title("Frozen downstream readouts", color=DARK)
    ax.grid(axis="y", color=GRID, lw=0.7)
    fig.suptitle(
        "Cross-domain and device transfer — full composite loss",
        fontsize=15,
        color=DARK,
        fontweight="bold",
    )
    save(fig, "external_scorecard")


def make_conditional_effect_map() -> None:
    frame = pd.read_csv(EFFECTS_PATH)
    frame = frame[
        frame.metric.eq("r2")
        & frame.effect_type.eq("main")
        & frame.effect.isin(["correlation", "mmd", "derivative"])
    ].copy()
    families = ["unet", "msvae", "ecgaim"]
    effects = ["correlation", "mmd", "derivative"]
    matrix = np.zeros((3, 3))
    labels = [["" for _ in effects] for _ in families]
    for i, family in enumerate(families):
        for j, effect in enumerate(effects):
            row = frame[frame.family.eq(family) & frame.effect.eq(effect)].iloc[0]
            matrix[i, j] = row.estimate
            labels[i][j] = (
                f"{row.estimate:+.4f}\n"
                f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
            )
    cmap = LinearSegmentedColormap.from_list(
        "embc_diverging", [GOLD, "#FFF8E8", "white", LIGHT_BLUE, BLUE]
    )
    limit = 0.03
    fig, ax = plt.subplots(figsize=(8.3, 3.25), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(3), ["Correlation", "MMD", "Derivative"])
    ax.set_yticks(range(3), ["U-Net", "MultiScale-VAE", "ECG-AIM"])
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(matrix[i, j]) > 0.018 else DARK,
                fontweight="bold" if abs(matrix[i, j]) > 0.01 else "normal",
            )
    ax.set_title(
        r"Conditional main effects on missing-lead $R^2$ (MSE-on slice)",
        color=DARK,
        fontsize=14,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"paired marginal $\Delta R^2$")
    save(fig, "conditional_r2_effects")


def write_summary_tables(master: pd.DataFrame) -> None:
    rows = []
    for family in ["unet", "multiscale_vae", "ecg_aim"]:
        subset = master[master.family.eq(family)]
        best = subset.loc[subset.ptbxl_missing_r2.idxmax()]
        full = subset[subset["mask"].eq("1111")].iloc[0]
        base = subset[subset["mask"].eq("1000")].iloc[0]
        rows.append(
            {
                "family": FAMILY_LABELS[family],
                "best_r2_mask": best["mask"],
                "best_r2": best.ptbxl_missing_r2,
                "full_pearson": full.ptbxl_missing_pearson,
                "full_qrs": full.ptbxl_qrs_correlation,
                "full_st": full.ptbxl_st_correlation,
                "full_ecgfounder_auroc": full.ecgfounder_150_macro_auroc,
                "delta_qrs_full_minus_mse": full.ptbxl_qrs_correlation
                - base.ptbxl_qrs_correlation,
                "delta_st_full_minus_mse": full.ptbxl_st_correlation
                - base.ptbxl_st_correlation,
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "poster_summary.csv", index=False)


def write_manifest() -> None:
    entries = []
    for path in sorted((OUT / "assets").rglob("*")):
        if path.is_file():
            entries.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "provenance": (
                        "Official EMBC 2026 PowerPoint template"
                        if "assets/logos" in str(path) or "assets/template" in str(path)
                        else "Generated from locked factorial_v4_2x4 results"
                    ),
                }
            )
    for path in sorted(DATA.glob("*")):
        if path.is_file():
            entries.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "provenance": "Locked comprehensive_latest_48_models evidence package",
                }
            )
    (OUT / "ASSET_MANIFEST.json").write_text(
        json.dumps({"schema": 1, "assets": entries}, indent=2) + "\n"
    )


def main() -> None:
    setup()
    copy_locked_inputs()
    master = read_master()
    make_factorial_atlas(master)
    make_clinical_effects()
    make_noise_small_multiples(master)
    make_external_scorecard(master)
    make_conditional_effect_map()
    # Real model output panel retained from the locked experiment.
    for suffix in ("pdf", "png"):
        source = (
            ROOT
            / "results"
            / "factorial_v4_2x4"
            / "plots"
            / f"echonext_representative_reconstructions.{suffix}"
        )
        shutil.copy2(source, FIG / f"echonext_representative_reconstructions.{suffix}")
    write_summary_tables(master)
    write_manifest()
    print(f"Generated poster assets in {OUT}")


if __name__ == "__main__":
    main()
