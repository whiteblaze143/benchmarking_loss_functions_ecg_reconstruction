"""Apply the frozen folds-1–6 Phase-KME map to fold 8 without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from repecg.common.kernels import NystromMap, WhiteningTransform
from scripts.paper02.build_paper02_representations import _atomic_npz, _eligible, _represent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.seed != 43:
        raise ValueError("fold-8 representation seed is frozen at 43")
    frame = _eligible(args.cache, (8,))
    with np.load(args.kernel_fit) as fit:
        whitening = WhiteningTransform(
            mean=np.asarray(fit["whitening_mean"]),
            components=np.asarray(fit["whitening_components"]),
            scales=np.asarray(fit["whitening_scales"]),
        )
        mapping = NystromMap(
            landmarks=np.asarray(fit["landmarks"]),
            inverse_root=np.asarray(fit["inverse_root"]),
            c2=float(fit["c2"]),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(args.output / "representation_pseudotest.npz", **_represent(args.cache, frame, whitening, mapping, args.seed))
    manifest = {
        "kind": "paper12_frozen_phase_kme_pseudotest",
        "fold": 8,
        "records": len(frame),
        "seed": args.seed,
        "kernel_fit_sha256": sha256(args.kernel_fit),
        "refit": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
