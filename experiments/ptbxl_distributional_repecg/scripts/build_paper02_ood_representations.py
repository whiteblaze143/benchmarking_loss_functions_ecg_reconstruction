from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd
import torch

from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import bandpass_ecg, independent_basis, LeadScaler
from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.paper02_kernel_mean.controls import exact_moment_matched_gaussian, mean_covariance_features
from tit_ecg.src.dataset_adapters import (
    EchoNextAdapter,
    EmoryAdapter,
    ISPAdapter,
    LUDBAdapter,
    RDBAdapter,
    SunnybrookAdapter,
    ZhejiangAdapter,
)

def _represent_ood(
    adapter,
    scaler: LeadScaler,
    whitening: WhiteningTransform,
    mapping: NystromMap,
    seed: int,
    batch_size: int = 128,
) -> dict[str, np.ndarray] | None:
    print(f"Building OOD representation for {adapter.name}...")
    records = adapter.list_records(max_records=50)
    if not records:
        print(f"No records found for {adapter.name}")
        return None

    beats_list = []
    labels_list = []
    ecg_ids = []

    for record_id in records:
        try:
            data = adapter.load_record(record_id, n_samples=3000)
            ecg = data["ecg"]
            if ecg.shape[1] != 12:
                print(f"Skipping {adapter.name} due to non-12-lead ECG shape: {ecg.shape}")
                return None
            
            fs = data["fs"]
            if fs != 500.0:
                from scipy.signal import resample_poly
                up = 500
                down = int(fs)
                ecg = resample_poly(ecg, up, down, axis=0)

            basis = independent_basis(bandpass_ecg(ecg, fs=500))
            detection = detect_rpeaks(basis)
            if len(detection.valid_intervals) < 3:
                continue
                
            beats = phase_normalize(scaler.transform(basis), detection.valid_intervals)
            if len(beats) < 3:
                continue

            beats_list.append(beats[:3])
            labels_list.append(np.zeros(5, dtype=np.float32))
            ecg_ids.append(record_id)
        except Exception as e:
            print(f"Error loading {record_id} from {adapter.name}: {e}")

    if not beats_list:
        return None

    torch.manual_seed(seed)
    device = torch.device("cuda")
    white_mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
    components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
    scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
    anchors = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
    inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)

    df = pd.DataFrame({
        "ecg_id": ecg_ids,
        "beats": beats_list,
        "labels": labels_list
    })
    df["valid_cycles"] = df["beats"].apply(len)
    
    count = len(df)
    kernel = np.zeros((count, 16, len(mapping.landmarks)), dtype=np.float32)
    gaussian = np.zeros_like(kernel)
    linear = np.zeros((count, 16, 8), dtype=np.float32)
    moments = np.zeros((count, 16, 44), dtype=np.float32)
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for valid_cycles, group in df.groupby("valid_cycles"):
        group_indices = group.index.to_numpy()
        for start in range(0, len(group_indices), batch_size):
            indices = group_indices[start : start + batch_size]
            b_count = len(indices)
            
            # Stack beats for this batch
            batch_beats = np.stack(group.loc[indices, "beats"].values)
            
            cells = torch.as_tensor(
                batch_beats.reshape(b_count, valid_cycles, 16, 16, 8)
                .transpose(0, 2, 1, 3, 4)
                .reshape(b_count * 16, valid_cycles * 16, 8),
                device=device,
            )
        with torch.inference_mode():
            matched = exact_moment_matched_gaussian(cells, generator=generator, tolerance=1e-5)

            def embed(values: torch.Tensor) -> torch.Tensor:
                white = (values.float() - white_mean) @ components * scales
                anchor_batch = anchors.expand(len(values), -1, -1)
                feature = torch.rsqrt(torch.cdist(white, anchor_batch).square() + mapping.c2)
                return (feature @ inverse_root).mean(dim=1)

            kernel[indices] = embed(cells).reshape(b_count, 16, -1).cpu().numpy()
            gaussian[indices] = embed(matched).reshape(b_count, 16, -1).cpu().numpy()
            linear[indices] = (
                ((cells.float() - white_mean) @ components * scales).mean(dim=1)
                .reshape(b_count, 16, 8)
                .cpu()
                .numpy()
            )
            moments[indices] = mean_covariance_features(cells.double()).reshape(b_count, 16, 44).float().cpu().numpy()

    return {
        "kernel": kernel,
        "gaussian": gaussian,
        "linear": linear,
        "moments": moments,
        "labels": np.stack(labels_list),
        "ecg_ids": np.array(ecg_ids, dtype=object),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.kernel_fit) as item:
        whitening = WhiteningTransform(
            mean=item["whitening_mean"],
            components=item["whitening_components"],
            scales=item["whitening_scales"],
        )
        mapping = NystromMap(
            landmarks=item["landmarks"],
            inverse_root=item["inverse_root"],
            c2=float(item["c2"]),
        )
        
    payload = json.loads(args.scaler.read_text())
    scaler = LeadScaler(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        std=np.asarray(payload["std"], dtype=np.float64),
    )

    adapters = [
        EchoNextAdapter(),
        EmoryAdapter(),
        ISPAdapter(),
        LUDBAdapter(),
        RDBAdapter(),
        SunnybrookAdapter(),
        ZhejiangAdapter(),
    ]

    for adapter in adapters:
        res = _represent_ood(adapter, scaler, whitening, mapping, args.seed)
        if res is not None:
            np.savez_compressed(args.output / f"representation_ood_{adapter.name.lower()}.npz", **res)
            print(f"Saved {adapter.name} representation.")

if __name__ == "__main__":
    main()
