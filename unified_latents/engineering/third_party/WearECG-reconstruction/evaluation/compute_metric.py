import os
import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

from ECGFounder import ft_ECGFounder
from metric_utils import (
    seed_everything,
    calculate_FID_score,
    calculate_frechet_distance,
    calculate_hr_score,
    calculate_mae,
    calculate_mse,
    calculate_statistics,
    compute_representations_in_batches,
    visualization,
    calculate_EMD_L2,
    calculate_CosDist,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ECG generation metrics")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="results/vae_100/MIMIC",
        help="Path to folder containing generated and real samples",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/metrics_results.csv",
        help="Path to output CSV file for saving metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="CUDA device index to use"
    )
    parser.add_argument(
        "--notes", type=str, default="", help="Additional notes for this run"
    )
    return parser.parse_args()


def extract_metadata_from_path(path):
    """Extract dataset and synthesis type from path"""
    path_parts = Path(path).parts
    return {
        "dataset": path_parts[-1],
        "synthesis_type": path_parts[-2] if len(path_parts) > 1 else "unknown",
    }


if __name__ == "__main__":
    args = parse_args()
    seed_everything(42, True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    device = args.device

    print(f"Computing Metric of {args.data_folder}")

    metadata = extract_metadata_from_path(args.data_folder)
    fake_samples = np.load(
        os.path.join(args.data_folder, "samples", "overall_fake_data.npy")
    )
    real_samples = np.load(
        os.path.join(args.data_folder, "samples", "ground_truth_5000_test.npy")
    )

    # representation model
    model = ft_ECGFounder(device=device)
    model.eval()
    # print(model)

    print(
        f"Original Data Shape: {real_samples.shape}, Generated Data Shape: {fake_samples.shape}"
    )
    print(
        f"Original Data Scale: [{np.min(real_samples)}, {np.max(real_samples)}] Generated Data Scale: [{np.min(fake_samples)}, {np.max(fake_samples)}]"
    )

    # Prepare results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": metadata["dataset"],
        "synthesis_type": metadata["synthesis_type"],
        "notes": args.notes,
    }

    # # compute MSE
    # mse_per_channel, overall_mse = calculate_mse(
    #     real_samples[:, :, 1:], fake_samples[:, :, 1:]
    # )
    # results["overall_mse"] = overall_mse
    # print("MSE per channel:", mse_per_channel)
    # print("Overall MSE (mean of all channels):", overall_mse)

    # # compute MAE
    # mae_per_channel, overall_mae = calculate_mae(
    #     real_samples[:, :, 1:], fake_samples[:, :, 1:]
    # )
    # results["overall_mae"] = overall_mae
    # print("MAE per channel:", mae_per_channel)
    # print("Overall MAE (mean of all channels):", overall_mae)

    # 定义要排除的通道索引
    # excluded_channels = [1, 6, 10]
    excluded_channels = [0, 1, 8]
    all_channels = list(range(real_samples.shape[2]))
    channels_to_keep = [i for i in all_channels if i not in excluded_channels]

    # 选择要保留的通道
    real_filtered = real_samples[:, :, channels_to_keep]
    fake_filtered = fake_samples[:, :, channels_to_keep]

    # compute MSE
    mse_per_channel, overall_mse = calculate_mse(real_filtered, fake_filtered)
    results["overall_mse"] = overall_mse
    print("MSE per channel:", mse_per_channel)
    print("Overall MSE (mean of all channels):", overall_mse)

    # compute MAE
    mae_per_channel, overall_mae = calculate_mae(real_filtered, fake_filtered)
    results["overall_mae"] = overall_mae
    print("MAE per channel:", mae_per_channel)
    print("Overall MAE (mean of all channels):", overall_mae)

    # emd_l2 = calculate_EMD_L2(real_samples[:, :, 1:], fake_samples[:, :, 1:])
    # print("Overall EMD L2:", emd_l2)
    # results["emd_l2"] = emd_l2

    # cos_distance = calculate_CosDist(real_samples[:, :, 1:], fake_samples[:, :, 1:])
    # print("Overall Cos Distance:", cos_distance)
    # results["cos_distance"] = cos_distance

    # compute FID
    fake_ecg_representation = compute_representations_in_batches(
        model, fake_samples, batch_size=128, device=device
    )
    real_ecg_representation = compute_representations_in_batches(
        model, real_samples, batch_size=128, device=device
    )
    fid = calculate_FID_score(fake_ecg_representation, real_ecg_representation)
    results["fid"] = fid
    print("FID: ", fid)

    # compute heart rate score
    # hr_score = calculate_hr_score(fake_samples, real_samples)
    # results["hr_score"] = hr_score
    # print("Heart rate score: ", hr_score)

    # 将通道级指标放在最后
    results["mse_per_channel"] = str(mse_per_channel)
    results["mae_per_channel"] = str(mae_per_channel)

    # Save results to CSV
    df = pd.DataFrame([results])
    if os.path.exists(args.output_csv):
        df.to_csv(args.output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")
