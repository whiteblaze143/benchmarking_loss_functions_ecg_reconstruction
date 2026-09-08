#!/usr/bin/env python3
"""
generate_external_cohort_figures.py
Generates publication-grade 12-lead ECG reconstruction gallery figures on external cohorts:
1. Sunnybrook Health Sciences Physical 10-Wire XMLs (ECG001.xml): 1-Lead and 3-Lead
2. Stanford EchoNext Structural Phenotype Cohort (Record #8): 1-Lead and 3-Lead
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sierraecg

_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)
from scripts.train_1lead_wavelet_ssl_mtl import build_model, forward_model
from scripts.evaluate_comprehensive_registry import load_adapter

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_GRID_POS = {
    "I": (0, 0), "II": (1, 0), "III": (2, 0),
    "aVR": (0, 1), "aVL": (1, 1), "aVF": (2, 1),
    "V1": (0, 2), "V2": (1, 2), "V3": (2, 2),
    "V4": (0, 3), "V5": (1, 3), "V6": (2, 3),
}

def draw_ecg_grid(ax, t_min=0.0, t_max=2.5, v_min=-1.5, v_max=1.5):
    t_minor = np.arange(t_min, t_max + 0.01, 0.04)
    for t in t_minor:
        is_major = np.isclose(t % 0.20, 0.0, atol=1e-3) or np.isclose(t % 0.20, 0.20, atol=1e-3)
        ax.axvline(t, color="#FFD1D1" if is_major else "#FFEBEB", linewidth=0.75 if is_major else 0.35, zorder=0)

    v_minor = np.arange(v_min, v_max + 0.01, 0.1)
    for v in v_minor:
        is_major = np.isclose(v % 0.5, 0.0, atol=1e-3) or np.isclose(v % 0.5, 0.5, atol=1e-3)
        ax.axhline(v, color="#FFD1D1" if is_major else "#FFEBEB", linewidth=0.75 if is_major else 0.35, zorder=0)

def pearson_r(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    return float(np.sum(vx * vy) / denom) if denom > 0 else 0.0

def plot_12lead_reconstruction(true_sig, pred_sig, observed_leads, title, subtitle, out_stem, v_range=(-1.5, 1.5)):
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.5), dpi=300, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.10, hspace=0.18, wspace=0.14)

    fs = 500
    n_samples = int(2.5 * fs) # 1250 samples
    time_axis = np.arange(n_samples) / fs

    r_dict = {}
    for idx, lead in enumerate(LEAD_NAMES):
        t_lead = true_sig[idx]
        p_lead = pred_sig[idx]
        r_dict[lead] = pearson_r(t_lead, p_lead)

    v_min, v_max = v_range

    for lead_idx, lead in enumerate(LEAD_NAMES):
        row, col = LEAD_GRID_POS[lead]
        ax = axes[row, col]
        draw_ecg_grid(ax, t_min=0.0, t_max=2.5, v_min=v_min, v_max=v_max)

        t_s = true_sig[lead_idx, :n_samples]
        p_s = pred_sig[lead_idx, :n_samples]

        is_obs = lead_idx in observed_leads
        if is_obs:
            ax.plot(time_axis, t_s, color="#2B6CB0", linewidth=1.5, label="Acquired Lead", zorder=3)
            status_text = "ACQUIRED"
            status_color = "#2B6CB0"
        else:
            ax.plot(time_axis, t_s, color="#1A202C", linewidth=1.2, label="Ground truth", zorder=2)
            ax.plot(time_axis, p_s, color="#E53E3E", linewidth=1.3, linestyle="--", label="ECG-AIM reconstruction", zorder=4)
            status_text = f"RECON (r={r_dict[lead]:.3f})"
            status_color = "#E53E3E"

        ax.text(0.04, 0.88, lead, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#1A365D", va="top")
        ax.text(0.04, 0.72, status_text, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=status_color, va="top")

        ax.set_xlim(0.0, 2.5)
        ax.set_ylim(v_min, v_max)

        if col == 0:
            ax.set_ylabel("Amplitude (mV)", fontsize=10, color="#1A202C")
            ax.tick_params(axis="y", labelsize=8.5, colors="#4A5568")
        else:
            ax.tick_params(axis="y", labelleft=False)

        if row == 2:
            ax.set_xlabel("Time (seconds)", fontsize=10, color="#1A202C")
            ax.set_xticks(np.arange(0.0, 2.6, 0.2))
            ax.tick_params(axis="x", labelsize=8.5, colors="#4A5568")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        for spine in ax.spines.values():
            spine.set_color("#CBD5E0")
            spine.set_linewidth(0.8)

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#1A365D", y=0.965)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=11.5, fontweight="semibold", color="#4A5568")

    obs_label = "Acquired Smartwatch Lead I" if len(observed_leads) == 1 else "Acquired Physical Triad [I, II, V2]"
    legend_elements = [
        Line2D([0], [0], color="#2B6CB0", lw=2, label=obs_label),
        Line2D([0], [0], color="#1A202C", lw=1.5, label="Ground truth"),
        Line2D([0], [0], color="#E53E3E", lw=1.5, linestyle="--", label="ECG-AIM reconstruction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E0", fontsize=10.5, bbox_to_anchor=(0.5, 0.02))

    for p in [_ROOT / f"figures/{out_stem}.png", _ROOT / f"slides/figures/{out_stem}.png"]:
        plt.savefig(p, dpi=300)
        print(f"Saved: {p}")
    for p in [_ROOT / f"figures/{out_stem}.pdf", _ROOT / f"slides/figures/{out_stem}.pdf"]:
        plt.savefig(p)
        print(f"Saved: {p}")
    plt.close(fig)

def main():
    print("Loading models...")
    ckpt_path = _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_wave_noSSL_gated_add_s42_l0/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])
    model_1l = build_model(cfg)
    model_1l.load_state_dict(ckpt["model_state_dict"], strict=True)
    model_1l.eval()

    spec_3l = {
        "id": "factorial_ecg_aim_1110000_s42",
        "kind": "alitok",
        "checkpoint": "checkpoints/cache/factorial_ecg_aim_1110000_s42.pt",
        "observed_leads": [0, 1, 7],
    }
    adapter_3l = load_adapter(spec_3l, torch.device("cpu"))

    # ----------------------------------------------------
    # 1. Sunnybrook Record ECG004 (Diagnostic amplitude)
    # ----------------------------------------------------
    print("\nProcessing Sunnybrook Record ECG004.xml...")
    import xml.etree.ElementTree as ET
    xml_path = _ROOT / "data/sunnybrook_12_lead_ecg_samples/ECG004.xml"
    tree = ET.parse(xml_path)
    pw = tree.getroot().find(".//{http://www3.medical.philips.com}parsedwaveforms")
    res_uv = float(pw.attrib.get("resolution", 5.0)) if pw is not None else 5.0
    f = sierraecg.read_file(str(xml_path))
    s_map = {l.label: l.samples for l in f.leads}
    sig_sb = np.stack([s_map[l] for l in LEAD_NAMES]).astype(np.float32)
    sig_sb = sig_sb[:, :5000] * (res_uv / 1000.0) # Physical mV
    tensor_sb = torch.from_numpy(sig_sb).unsqueeze(0)

    # 1-Lead Sunnybrook
    with torch.no_grad():
        res_sb_1l = forward_model(model_1l, tensor_sb, [0], compute_delineation=False, compute_ssl=False)
        pred_sb_1l = res_sb_1l["y_pred"][0].numpy()
        pred_sb_1l[0] = sig_sb[0]
    r_sb_1l = [pearson_r(sig_sb[i], pred_sb_1l[i]) for i in range(1, 12)]
    mean_r_sb_1l = float(np.mean(r_sb_1l))

    plot_12lead_reconstruction(
        sig_sb, pred_sb_1l, [0],
        "ECG-AIM 1-Lead Reconstruction: Sunnybrook Physical 10-Wire XML (ECG004)",
        f"Acquired Lead I (Smartwatch Vector) $\\rightarrow$ 11 Missing Leads Inferred (Record Mean Missing-Lead r = {mean_r_sb_1l:.3f})",
        "sunnybrook_best_reconstruction_1lead",
        v_range=(-1.5, 1.5)
    )

    # 3-Lead Sunnybrook
    with torch.no_grad():
        pred_sb_3l = adapter_3l.reconstruct(tensor_sb)[0].numpy()
        pred_sb_3l[[0, 1, 7]] = sig_sb[[0, 1, 7]]
    missing_3l = [i for i in range(12) if i not in [0, 1, 7]]
    r_sb_3l = [pearson_r(sig_sb[i], pred_sb_3l[i]) for i in missing_3l]
    mean_r_sb_3l = float(np.mean(r_sb_3l))

    plot_12lead_reconstruction(
        sig_sb, pred_sb_3l, [0, 1, 7],
        "ECG-AIM 3-Lead Reconstruction: Sunnybrook Physical 10-Wire XML (ECG004)",
        f"Acquired Physical Triad [I, II, V2] $\\rightarrow$ 9 Missing Leads Inferred (Record Mean Missing-Lead r = {mean_r_sb_3l:.3f})",
        "sunnybrook_best_reconstruction_3lead",
        v_range=(-1.5, 1.5)
    )

    # ----------------------------------------------------
    # 2. EchoNext Record #10 (Diagnostic amplitude)
    # ----------------------------------------------------
    print("\nProcessing EchoNext Record #10...")
    provenance = {
        "dataset": "EchoNext",
        "units": "uV",
        "normalization": {
            "kind": "dataset_zscore",
            "mean": [5.438902932045533, 4.386144027595723, -0.8828521117626768, -4.901228423594343, 3.179303315626078, 1.718962497412901, -4.3853678923766815, -1.4991789389444636, -0.2390101635046568, 3.622168549154881, 5.396823627457744, 5.660982479475681],
            "std": [32.4283237955487, 31.37262367036232, 30.155474930762434, 28.126380180199195, 27.097711990189254, 26.148743373957526, 37.739210938823206, 52.792504469040814, 56.75138860280555, 49.90223860158139, 45.001427330308026, 39.06606103398455]
        }
    }
    from scripts.evaluate_echonext import EchoNextWaveforms
    raw_data = np.load(_ROOT / "data/echonext/EchoNext_test_waveforms.npy", mmap_mode="r")
    loader = EchoNextWaveforms(raw_data, provenance)
    batch = loader.batch(10, 11) # Record 10
    sig_en = batch[0] * 5.0 # Physical mV (5 uV / count resolution)
    tensor_en = torch.from_numpy(sig_en).unsqueeze(0)

    # 1-Lead EchoNext
    with torch.no_grad():
        res_en_1l = forward_model(model_1l, tensor_en, [0], compute_delineation=False, compute_ssl=False)
        pred_en_1l = res_en_1l["y_pred"][0].numpy()
        pred_en_1l[0] = sig_en[0]
    r_en_1l = [pearson_r(sig_en[i], pred_en_1l[i]) for i in range(1, 12)]
    mean_r_en_1l = float(np.mean(r_en_1l))

    plot_12lead_reconstruction(
        sig_en, pred_en_1l, [0],
        "ECG-AIM 1-Lead Reconstruction: Stanford EchoNext Cohort (Record #10)",
        f"Acquired Lead I (Smartwatch Vector) $\\rightarrow$ 11 Missing Leads Inferred (Record Mean Missing-Lead r = {mean_r_en_1l:.3f})",
        "echonext_best_reconstruction_1lead",
        v_range=(-1.5, 1.5)
    )

    # 3-Lead EchoNext
    with torch.no_grad():
        pred_en_3l = adapter_3l.reconstruct(tensor_en)[0].numpy()
        pred_en_3l[[0, 1, 7]] = sig_en[[0, 1, 7]]
    r_en_3l = [pearson_r(sig_en[i], pred_en_3l[i]) for i in missing_3l]
    mean_r_en_3l = float(np.mean(r_en_3l))

    plot_12lead_reconstruction(
        sig_en, pred_en_3l, [0, 1, 7],
        "ECG-AIM 3-Lead Reconstruction: Stanford EchoNext Cohort (Record #10)",
        f"Acquired Physical Triad [I, II, V2] $\\rightarrow$ 9 Missing Leads Inferred (Record Mean Missing-Lead r = {mean_r_en_3l:.3f})",
        "echonext_best_reconstruction_3lead",
        v_range=(-1.5, 1.5)
    )

    print("\nAll 4 figures generated successfully!")

if __name__ == "__main__":
    main()
