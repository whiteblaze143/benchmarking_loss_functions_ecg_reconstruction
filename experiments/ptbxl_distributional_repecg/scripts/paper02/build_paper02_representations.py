from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.common.kernels import NystromMap, WhiteningTransform, audit_nystrom, biased_mmd2
from repecg.paper02_kernel_mean.controls import exact_moment_matched_gaussian, mean_covariance_features


def _eligible(cache: Path, folds: tuple[int, ...] | None = None) -> pd.DataFrame:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["kind"] != "production_phase_cache":
        raise ValueError(f"not a production cache: {cache}")
    frame = pd.read_csv(cache / "qc.csv")
    frame = frame[frame.eligible]
    if folds is not None:
        frame = frame[frame.fold.isin(folds)]
    return frame.reset_index(drop=True)


def _load_beats(cache: Path, artifact: str) -> np.ndarray:
    with np.load(cache / artifact) as item:
        beats = np.asarray(item["beats"], dtype=np.float64)
    if beats.ndim != 3 or beats.shape[1:] != (256, 8):
        raise ValueError(f"invalid beat tensor in {artifact}: {beats.shape}")
    return beats


def _phase_atoms(beats: np.ndarray) -> list[np.ndarray]:
    cells = beats.reshape(len(beats), 16, 16, 8)
    return [cells[:, phase].reshape(-1, 8) for phase in range(16)]


def _reservoir(cache: Path, frame: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    chunks: list[np.ndarray] = []
    observed = 0
    for index in order:
        atoms = _load_beats(cache, str(frame.iloc[index].artifact)).reshape(-1, 8)
        if observed + len(atoms) > count:
            atoms = atoms[rng.choice(len(atoms), size=count - observed, replace=False)]
        chunks.append(atoms)
        observed += len(atoms)
        if observed == count:
            break
    if observed < count:
        raise ValueError(f"only {observed} training atoms available; requested {count}")
    return np.concatenate(chunks)


def _audit_cells(
    train_cache: Path,
    train: pd.DataFrame,
    select_cache: Path,
    select: pd.DataFrame,
    pairs: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    cells: list[np.ndarray] = []
    for cache, frame in ((train_cache, train), (select_cache, select)):
        take = min(len(frame), max(64, pairs // 8))
        for index in rng.choice(len(frame), size=take, replace=False):
            beats = _load_beats(cache, str(frame.iloc[index].artifact))
            cells.extend(_phase_atoms(beats))
    rng.shuffle(cells)
    return cells


def _audit(
    cells: list[np.ndarray],
    whitening: WhiteningTransform,
    mapping: NystromMap,
    pairs: int,
    seed: int,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
    components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
    scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
    anchors = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
    inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)
    exact = np.empty(pairs, dtype=np.float64)
    approximate = np.empty(pairs, dtype=np.float64)
    with torch.inference_mode():
        for pair in range(pairs):
            left, right = rng.choice(len(cells), size=2, replace=False)
            x = (torch.as_tensor(cells[left], device=device, dtype=torch.float32) - mean) @ components * scales
            y = (torch.as_tensor(cells[right], device=device, dtype=torch.float32) - mean) @ components * scales
            exact[pair] = float(
                torch.rsqrt(torch.cdist(x, x).square() + mapping.c2).mean()
                + torch.rsqrt(torch.cdist(y, y).square() + mapping.c2).mean()
                - 2.0 * torch.rsqrt(torch.cdist(x, y).square() + mapping.c2).mean()
            )
            x_feature = torch.rsqrt(torch.cdist(x, anchors).square() + mapping.c2) @ inverse_root
            y_feature = torch.rsqrt(torch.cdist(y, anchors).square() + mapping.c2) @ inverse_root
            approximate[pair] = float(torch.square(x_feature.mean(0) - y_feature.mean(0)).sum())
    result = audit_nystrom(exact, approximate)
    positive = exact[exact > 0]
    relative_error_floor = max(1e-8, 0.001 * float(np.median(positive)))
    relative_error = np.abs(approximate - exact) / np.maximum(np.abs(exact), relative_error_floor)
    result["relative_error_floor"] = relative_error_floor
    result["relative_error_q90"] = float(np.quantile(relative_error, 0.90))
    result["relative_error_q95"] = float(np.quantile(relative_error, 0.95))
    result["relative_error_q99"] = float(np.quantile(relative_error, 0.99))
    return result


def _represent(
    cache: Path,
    frame: pd.DataFrame,
    whitening: WhiteningTransform,
    mapping: NystromMap,
    seed: int,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    white_mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
    components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
    scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
    anchors = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
    inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)
    kernel = np.empty((len(frame), 16, len(mapping.landmarks)), dtype=np.float32)
    gaussian = np.empty_like(kernel)
    linear = np.empty((len(frame), 16, 8), dtype=np.float32)
    moments = np.empty((len(frame), 16, 44), dtype=np.float32)
    labels = np.empty((len(frame), 5), dtype=np.float32)
    ecg_ids = frame.ecg_id.to_numpy(dtype=np.int64)
    patient_ids = frame.patient_id.to_numpy(dtype=np.int64)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    def load(index: int) -> tuple[int, np.ndarray, np.ndarray]:
        row = frame.iloc[index]
        with np.load(cache / str(row.artifact)) as item:
            return index, np.asarray(item["beats"], dtype=np.float32), np.asarray(item["labels"], dtype=np.float32)

    with ThreadPoolExecutor(max_workers=8) as loader:
        for _, group in frame.groupby("valid_cycles", sort=False):
            group_indices = group.index.to_numpy(dtype=np.int64)
            for start in range(0, len(group_indices), batch_size):
                indices = group_indices[start : start + batch_size]
                payload = list(loader.map(load, indices))
                beat_batch = np.stack([item[1] for item in payload])
                labels[indices] = np.stack([item[2] for item in payload])
                count, beats_per_record = beat_batch.shape[:2]
                cells = torch.as_tensor(
                    beat_batch.reshape(count, beats_per_record, 16, 16, 8)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(count * 16, beats_per_record * 16, 8),
                    device=device,
                )
                with torch.inference_mode():
                    matched = exact_moment_matched_gaussian(cells, generator=generator)

                    def embed(values: torch.Tensor) -> torch.Tensor:
                        white = (values.float() - white_mean) @ components * scales
                        anchor_batch = anchors.expand(len(values), -1, -1)
                        feature = torch.rsqrt(torch.cdist(white, anchor_batch).square() + mapping.c2)
                        return (feature @ inverse_root).mean(dim=1)

                    kernel[indices] = embed(cells).reshape(count, 16, -1).cpu().numpy()
                    gaussian[indices] = embed(matched).reshape(count, 16, -1).cpu().numpy()
                    linear[indices] = (
                        ((cells - white_mean) @ components * scales).mean(dim=1)
                        .reshape(count, 16, 8)
                        .cpu()
                        .numpy()
                    )
                    moments[indices] = mean_covariance_features(cells.double()).reshape(count, 16, 44).float().cpu().numpy()
    return {
        "kernel": kernel,
        "gaussian": gaussian,
        "linear": linear,
        "moments": moments,
        "labels": labels,
        "ecg_ids": ecg_ids,
        "patient_ids": patient_ids,
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--select-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir", type=int, default=500_000)
    parser.add_argument("--audit-pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-folds", type=int, nargs="+")
    parser.add_argument("--select-folds", type=int, nargs="+")
    parser.add_argument("--landmark-candidates", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--min-spearman", type=float, default=0.90)
    parser.add_argument("--max-median-relative-error", type=float, default=0.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = _eligible(args.train_cache, None if args.train_folds is None else tuple(args.train_folds))
    select = _eligible(args.select_cache, None if args.select_folds is None else tuple(args.select_folds))
    if train.empty or select.empty:
        raise ValueError("fold filtering produced an empty representation split")
    reservoir = _reservoir(args.train_cache, train, args.reservoir, args.seed)
    whitening = WhiteningTransform.fit(reservoir)
    whitened = whitening.transform(reservoir)
    cells = _audit_cells(args.train_cache, train, args.select_cache, select, args.audit_pairs, args.seed)
    audits: dict[str, dict[str, float | bool]] = {}
    mapping: NystromMap | None = None
    for landmarks in args.landmark_candidates:
        mapping = NystromMap.fit(whitened, landmarks=landmarks, c2=1.0, seed=args.seed)
        result = _audit(cells, whitening, mapping, args.audit_pairs, args.seed)
        result["min_spearman"] = args.min_spearman
        result["max_median_relative_error"] = args.max_median_relative_error
        result["passed"] = bool(
            result["spearman"] >= args.min_spearman
            and result["median_relative_error"] < args.max_median_relative_error
        )
        audits[str(landmarks)] = result
        if result["passed"]:
            break
    assert mapping is not None
    audit_path = args.output / "nystrom_audit.json"
    audit_path.write_text(json.dumps({"pairs": args.audit_pairs, "c2": 1.0, "audits": audits}, indent=2, sort_keys=True) + "\n")
    if not audits[str(len(mapping.landmarks))]["passed"]:
        raise RuntimeError(f"Nyström fidelity gate failed: {audits}")
    _atomic_npz(
        args.output / "kernel_fit.npz",
        whitening_mean=whitening.mean,
        whitening_components=whitening.components,
        whitening_scales=whitening.scales,
        landmarks=mapping.landmarks,
        inverse_root=mapping.inverse_root,
        c2=np.asarray(mapping.c2),
    )
    _atomic_npz(
        args.output / "representation_train.npz",
        **_represent(args.train_cache, train, whitening, mapping, args.seed),
    )
    _atomic_npz(
        args.output / "representation_val.npz",
        **_represent(args.select_cache, select, whitening, mapping, args.seed + 1),
    )
    manifest = {
        "kind": "paper02_development_representations",
        "train_records": len(train),
        "validation_records": len(select),
        "landmarks": len(mapping.landmarks),
        "audit": audits[str(len(mapping.landmarks))],
        "representations": {
            "kernel": [16, len(mapping.landmarks)],
            "linear": [16, 8],
            "moments": [16, 44],
            "gaussian": [16, len(mapping.landmarks)],
        },
        "controls": {
            "moments": "mean_plus_lower_triangle_sample_covariance",
            "gaussian": "realized_mean_and_sample_covariance_matched",
            "moment_tolerance": 1e-8,
            "linear": "mean_in_train_whitened_linear_kernel_space",
        },
        "seed": args.seed,
        "folds": {"train": args.train_folds, "selection": args.select_folds},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
