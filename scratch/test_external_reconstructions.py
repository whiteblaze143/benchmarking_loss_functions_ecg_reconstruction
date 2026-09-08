#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sierraecg

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)
from scripts.train_1lead_wavelet_ssl_mtl import build_model, forward_model
from scripts.evaluate_comprehensive_registry import load_adapter

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def pearson_r(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    return float(np.sum(vx * vy) / denom) if denom > 0 else 0.0

def test_sunnybrook():
    print("--- Testing Sunnybrook Records ---")
    xml_files = sorted(list((_ROOT / "data/sunnybrook_12_lead_ecg_samples").glob("*.xml")))
    print(f"Found {len(xml_files)} Sunnybrook XMLs.")

    # Load 1-lead champion
    ckpt_path = _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_wave_noSSL_gated_add_s42_l0/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    import argparse
    cfg = argparse.Namespace(**ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Load 3-lead adapter
    spec_3l = {
        "id": "factorial_ecg_aim_1110000_s42",
        "kind": "alitok",
        "checkpoint": "checkpoints/cache/factorial_ecg_aim_1110000_s42.pt",
        "observed_leads": [0, 1, 7],
    }
    adapter_3l = load_adapter(spec_3l, torch.device("cpu"))

    results = []
    for xml_f in xml_files:
        f = sierraecg.read_file(str(xml_f))
        s_map = {l.label: l.samples for l in f.leads}
        sig = np.stack([s_map[l] for l in LEAD_NAMES]).astype(np.float32)
        # Resample or trim to 5000 samples (10s @ 500Hz)
        if sig.shape[1] > 5000:
            sig = sig[:, :5000]
        elif sig.shape[1] < 5000:
            sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])))
        
        # Convert uV to mV
        sig_mv = sig / 1000.0
        tensor = torch.from_numpy(sig_mv).unsqueeze(0) # [1, 12, 5000]

        # 1-Lead (Lead I: index 0)
        with torch.no_grad():
            res_1l = forward_model(model, tensor, [0], compute_delineation=False, compute_ssl=False)
            pred_1l = res_1l["y_pred"][0].numpy()
            pred_1l[0] = sig_mv[0]
            r_1l = [pearson_r(sig_mv[i], pred_1l[i]) for i in range(1, 12)]
            mean_r_1l = float(np.mean(r_1l))

        # 3-Lead (Leads 0, 1, 7)
        with torch.no_grad():
            pred_3l = adapter_3l.reconstruct(tensor)[0].numpy()
            pred_3l[[0, 1, 7]] = sig_mv[[0, 1, 7]]
            missing_3l = [i for i in range(12) if i not in [0, 1, 7]]
            r_3l = [pearson_r(sig_mv[i], pred_3l[i]) for i in missing_3l]
            mean_r_3l = float(np.mean(r_3l))

        results.append((xml_f.name, mean_r_1l, mean_r_3l))
        print(f"{xml_f.name}: 1-Lead mean r = {mean_r_1l:.3f}, 3-Lead mean r = {mean_r_3l:.3f}")

    best_sb = max(results, key=lambda x: x[1] + x[2])
    print(f"\nBest Sunnybrook Record: {best_sb}")

def test_echonext():
    print("\n--- Testing EchoNext Records ---")
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
    batch = loader.batch(0, 30) # 30 records [30, 12, 5000] in mV

    ckpt_path = _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_wave_noSSL_gated_add_s42_l0/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    import argparse
    cfg = argparse.Namespace(**ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    spec_3l = {
        "id": "factorial_ecg_aim_1110000_s42",
        "kind": "alitok",
        "checkpoint": "checkpoints/cache/factorial_ecg_aim_1110000_s42.pt",
        "observed_leads": [0, 1, 7],
    }
    adapter_3l = load_adapter(spec_3l, torch.device("cpu"))

    results = []
    for idx in range(batch.shape[0]):
        sig_mv = batch[idx] # [12, 5000]
        tensor = torch.from_numpy(sig_mv).unsqueeze(0)

        with torch.no_grad():
            res_1l = forward_model(model, tensor, [0], compute_delineation=False, compute_ssl=False)
            pred_1l = res_1l["y_pred"][0].numpy()
            pred_1l[0] = sig_mv[0]
            r_1l = [pearson_r(sig_mv[i], pred_1l[i]) for i in range(1, 12)]
            mean_r_1l = float(np.mean(r_1l))

            pred_3l = adapter_3l.reconstruct(tensor)[0].numpy()
            pred_3l[[0, 1, 7]] = sig_mv[[0, 1, 7]]
            missing_3l = [i for i in range(12) if i not in [0, 1, 7]]
            r_3l = [pearson_r(sig_mv[i], pred_3l[i]) for i in missing_3l]
            mean_r_3l = float(np.mean(r_3l))

        results.append((idx, mean_r_1l, mean_r_3l))
        print(f"EchoNext Record #{idx}: 1-Lead mean r = {mean_r_1l:.3f}, 3-Lead mean r = {mean_r_3l:.3f}")

    best_en = max(results, key=lambda x: x[1] + x[2])
    print(f"\nBest EchoNext Record: {best_en}")

if __name__ == "__main__":
    test_sunnybrook()
    test_echonext()
