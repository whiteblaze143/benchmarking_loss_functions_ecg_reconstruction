from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.common.bootstrap import interval
from repecg.paper02_kernel_mean.controls import mean_covariance_features


def _similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Zero-to-one normalized Frobenius similarity with explicit zero cases."""
    left = left.flatten(1)
    right = right.flatten(1)
    numerator = 2.0 * (left * right).sum(dim=1)
    denominator = left.square().sum(dim=1) + right.square().sum(dim=1)
    both_zero = denominator == 0
    value = torch.where(both_zero, torch.ones_like(denominator), numerator / denominator.clamp_min(1e-12))
    return value.clamp(0.0, 1.0)


def _patient_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> np.ndarray:
    patient = frame.groupby("patient_id", sort=False)["delta_kernel_minus_moments"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    return np.asarray(
        [patient[rng.integers(0, len(patient), len(patient))].mean() for _ in range(replicates)],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    qc = pd.read_csv(args.cache / "qc.csv")
    primary_eligible = qc[qc.eligible]
    eligible = primary_eligible[primary_eligible.valid_cycles >= 4].reset_index(drop=True)
    with np.load(args.kernel_fit) as fit:
        white_mean = torch.as_tensor(fit["whitening_mean"], device="cuda", dtype=torch.float32)
        components = torch.as_tensor(fit["whitening_components"], device="cuda", dtype=torch.float32)
        scales = torch.as_tensor(fit["whitening_scales"], device="cuda", dtype=torch.float32)
        anchors = torch.as_tensor(fit["landmarks"], device="cuda", dtype=torch.float32)
        inverse_root = torch.as_tensor(fit["inverse_root"], device="cuda", dtype=torch.float32)
        c2 = float(fit["c2"])

    def load(index: int) -> tuple[int, np.ndarray]:
        with np.load(args.cache / str(eligible.iloc[index].artifact)) as item:
            return index, np.asarray(item["beats"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    torch.backends.cudnn.enabled = False
    with ThreadPoolExecutor(max_workers=8) as loader:
        for _, group in eligible.groupby("valid_cycles", sort=False):
            indices = group.index.to_numpy(dtype=np.int64)
            for start in range(0, len(indices), args.batch):
                batch_indices = indices[start : start + args.batch]
                payload = list(loader.map(load, batch_indices))
                beat_batch = np.stack([item[1] for item in payload])
                count, beats_per_record = beat_batch.shape[:2]

                representations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                for name, positions in (("odd", np.arange(0, beats_per_record, 2)), ("even", np.arange(1, beats_per_record, 2))):
                    selected = beat_batch[:, positions]
                    cells = torch.as_tensor(
                        selected.reshape(count, len(positions), 16, 16, 8)
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(count * 16, len(positions) * 16, 8),
                        device="cuda",
                    )
                    with torch.inference_mode():
                        white = (cells - white_mean) @ components * scales
                        features = torch.rsqrt(
                            torch.cdist(white, anchors.expand(len(white), -1, -1)).square() + c2
                        ) @ inverse_root
                        kernel = features.mean(dim=1).reshape(count, 16, -1)
                        moments = mean_covariance_features(cells.double()).float().reshape(count, 16, 44)
                    representations[name] = (kernel, moments)

                kernel_similarity = _similarity(representations["odd"][0], representations["even"][0]).cpu().numpy()
                moment_similarity = _similarity(representations["odd"][1], representations["even"][1]).cpu().numpy()
                for offset, frame_index in enumerate(batch_indices):
                    row = eligible.iloc[frame_index]
                    rows.append(
                        {
                            "ecg_id": int(row.ecg_id),
                            "patient_id": int(row.patient_id),
                            "valid_cycles": int(row.valid_cycles),
                            "kernel_similarity": float(kernel_similarity[offset]),
                            "moments_similarity": float(moment_similarity[offset]),
                            "delta_kernel_minus_moments": float(kernel_similarity[offset] - moment_similarity[offset]),
                        }
                    )

    result = pd.DataFrame(rows).sort_values("ecg_id")
    result.to_csv(args.output / "odd_even_stability.csv", index=False)
    draws = _patient_bootstrap(result, args.replicates, args.seed)
    np.save(args.output / "odd_even_stability_bootstrap.npy", draws)
    patient_means = result.groupby("patient_id", sort=False).mean(numeric_only=True)
    summary = {
        "kind": "paper02_development_odd_even_stability",
        "metric": "clipped_normalized_frobenius_similarity",
        "records": int(len(result)),
        "patients": int(result.patient_id.nunique()),
        "requested_records": int(len(qc)),
        "primary_eligible_records": int(len(primary_eligible)),
        "primary_ineligible_records": int(len(qc) - len(primary_eligible)),
        "additional_ineligible_fewer_than_four_cycles": int(len(primary_eligible) - len(result)),
        "kernel_patient_equal": float(patient_means.kernel_similarity.mean()),
        "moments_patient_equal": float(patient_means.moments_similarity.mean()),
        "delta_patient_equal": float(patient_means.delta_kernel_minus_moments.mean()),
        "delta_ci95": interval(draws),
        "replicates": args.replicates,
        "seed": args.seed,
    }
    (args.output / "odd_even_stability.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
