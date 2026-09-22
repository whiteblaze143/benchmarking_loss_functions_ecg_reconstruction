#!/usr/bin/env python3
"""Non-proxy ECGFounder representation evaluation on fixed EchoNext test data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(repo: Path, checkpoint_path: Path, leads: int, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    from net1d import Net1D
    model = Net1D(
        in_channels=leads, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16,
        stride=2, groups_width=16, n_classes=150, use_bn=False,
        use_do=False, return_features=True, verbose=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("checkpoint lacks state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def preprocess(batch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert released EchoNext scaling, resample 250->500 Hz, then author filtering/z-score."""
    # The source release is 250 Hz. Resampling precedes the official 500-Hz filter
    # solely to satisfy the filter's declared sampling-rate contract.
    restored = batch * std[None, :, None] + mean[None, :, None]
    upsampled = resample_poly(restored, up=2, down=1, axis=-1).astype(np.float32)
    from util import filter_bandpass
    filtered = np.stack([filter_bandpass(record, 500) for record in upsampled])
    return (filtered - filtered.mean(axis=(1, 2), keepdims=True)) / (
        filtered.std(axis=(1, 2), keepdims=True) + 1e-8
    )


def probe_cv(z: np.ndarray, y: np.ndarray) -> dict[str, object]:
    fold_scores: list[float] = []
    folds: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(z)):
        scaler = StandardScaler().fit(z[train_idx])
        z_train, z_test = scaler.transform(z[train_idx]), scaler.transform(z[test_idx])
        per_label: list[float] = []
        for label in range(y.shape[1]):
            if len(np.unique(y[train_idx, label])) != 2 or len(np.unique(y[test_idx, label])) != 2:
                raise RuntimeError(f"EchoNext fold {fold}, label {label} lacks both classes")
            classifier = LogisticRegression(max_iter=500, C=1.0)
            classifier.fit(z_train, y[train_idx, label])
            per_label.append(float(roc_auc_score(y[test_idx, label], classifier.predict_proba(z_test)[:, 1])))
        folds.append({"fold": fold, "macro_auroc": float(np.mean(per_label)), "classwise_auroc": per_label})
        fold_scores.append(float(np.mean(per_label)))
    return {"macro_auroc": float(np.mean(fold_scores)), "cv_aurocs": fold_scores, "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--echonext-dir", type=Path, required=True)
    parser.add_argument("--leads", type=int, choices=(1, 12), required=True)
    parser.add_argument("--lead-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.leads == 12 and args.lead_index != 0:
        raise ValueError("--lead-index is only valid with --leads 1")

    provenance = json.loads((args.echonext_dir / "PROVENANCE.json").read_text())
    required = {"sampling_rate_hz": 250, "num_leads": 12, "samples_per_lead": 2500}
    for key, value in required.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"unexpected EchoNext provenance {key}: {provenance.get(key)!r}")
    if provenance.get("normalization", {}).get("kind") != "dataset_zscore":
        raise RuntimeError("cannot invert an unknown EchoNext normalization")
    if provenance.get("lead_order", [None])[args.lead_index] != "I" and args.leads == 1:
        raise RuntimeError("single-lead run is restricted to physical Lead I")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repecg.evaluation.native_labels import ECHONEXT_LABELS, load_echonext_labels
    waveforms = np.load(args.echonext_dir / "EchoNext_test_waveforms.npy", mmap_mode="r")
    labels = load_echonext_labels(args.echonext_dir / "echonext_metadata_100k.csv", split="test", expected_records=len(waveforms))
    y = labels.loc[:, ECHONEXT_LABELS].to_numpy(dtype=np.float32)
    mean = np.asarray(provenance["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(provenance["normalization"]["std"], dtype=np.float32)
    device = torch.device("cuda")
    model = make_model(args.repo, args.checkpoint, args.leads, device)

    chunks: list[np.ndarray] = []
    for start in range(0, len(waveforms), args.batch_size):
        raw = np.transpose(waveforms[start:start + args.batch_size, 0], (0, 2, 1)).astype(np.float32)
        prepared = preprocess(raw, mean, std)
        if args.leads == 1:
            prepared = prepared[:, args.lead_index:args.lead_index + 1]
        with torch.no_grad():
            _, features = model(torch.from_numpy(prepared).to(device))
        chunks.append(features.cpu().numpy())
    z = np.concatenate(chunks)
    scores = probe_cv(z, y)

    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "embeddings.npz", embeddings=z, labels=y, record_index=np.arange(len(z)))
    payload = {
        "dataset": "echonext", "task_id": "echonext_shd_12", "n_samples": int(len(z)),
        "embedding_dim": int(z.shape[1]), "model": f"ecgfounder_{args.leads}lead",
        "lead_contract": "full_12_leads" if args.leads == 12 else "physical_lead_I_only",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
        "echonext_provenance": provenance, "preprocessing": "invert_dataset_zscore_then_250_to_500_resample_then_official_500Hz_filter_then_per_record_zscore",
        **scores,
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
