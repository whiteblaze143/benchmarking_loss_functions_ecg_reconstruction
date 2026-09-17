#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from repecg.common.kernels import TruncatedPCAWhitening


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper02-source", type=Path, required=True)
    parser.add_argument("--paper03-output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.paper02_source / "representation_train.npz") as item:
        source = np.asarray(item["kernel"], dtype=np.float64)
    whitening = TruncatedPCAWhitening.fit(
        source.reshape(-1, source.shape[-1]), n_components=128
    )
    if len(whitening.scales) != 128:
        raise ValueError(f"unordered control retained {len(whitening.scales)} components, expected 128")
    _atomic_npz(
        args.paper03_output / "unordered_control_fit.npz",
        mean=whitening.mean,
        components=whitening.components,
        scales=whitening.scales,
    )
    for split in ("train", "val"):
        paper03_path = args.paper03_output / f"representation_{split}.npz"
        with np.load(paper03_path) as item:
            arrays = {name: np.asarray(item[name]) for name in item.files}
        with np.load(args.paper02_source / f"representation_{split}.npz") as item:
            source = np.asarray(item["kernel"], dtype=np.float64)
        arrays["unordered_kme"] = whitening.transform(
            source.reshape(-1, source.shape[-1])
        ).reshape(source.shape[0], 16, 128).astype(np.float32)
        _atomic_npz(paper03_path, **arrays)
    manifest_path = args.paper03_output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unordered_control_components"] = 128
    manifest["unordered_control_fit"] = "folds_1_7_truncated_pca_whitening"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "unordered_control_components": 128}))


if __name__ == "__main__":
    main()
