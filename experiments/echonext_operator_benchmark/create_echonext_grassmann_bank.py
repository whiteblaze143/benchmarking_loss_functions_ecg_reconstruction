#!/usr/bin/env python3
"""Construct and freeze the official EchoNext Grassmann-350 Subspace Bank.

Constructs 350 frozen subspaces:
- 50 rank-1 subspaces in Gr(1, 8)
- 50 rank-2 subspaces in Gr(2, 8)
- 50 rank-3 subspaces in Gr(3, 8)
- 50 rank-4 subspaces in Gr(4, 8)
- 50 rank-5 subspaces in Gr(5, 8)
- 50 rank-6 subspaces in Gr(6, 8)
- 50 rank-7 subspaces in Gr(7, 8)

Each subspace B_U in R^{r x 8} satisfies strict orthonormality: B_U B_U^T = I_r.
Saves:
- /data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results/echonext_grassmann_350_bank.npz
- /data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results/echonext_grassmann_350_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_echonext_grassmann_bank")

OUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/echonext_three_way_v1/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def generate_bank(seed: int = 2026, n_per_rank: int = 50) -> Path:
    bank_path = OUT_DIR / "echonext_grassmann_350_bank.npz"
    manifest_path = OUT_DIR / "echonext_grassmann_350_manifest.json"

    logger.info("Generating EchoNext Grassmann-350 Subspace Bank (seed=%d)...", seed)
    rng = np.random.default_rng(seed)
    bank_data = {}

    for r in range(1, 8):
        subspaces = []
        for s_idx in range(n_per_rank):
            G = rng.normal(size=(8, r))
            Q, _ = np.linalg.qr(G)
            B_U = Q.T.astype(np.float32)  # (r, 8)

            # Strict orthonormality verification
            gram = B_U @ B_U.T
            err = float(np.max(np.abs(gram - np.eye(r, dtype=np.float32))))
            assert err < 1e-5, f"Orthonormality violation at rank {r}, sub {s_idx}: {err}"

            subspaces.append(B_U)
        bank_data[f"rank_{r}"] = np.stack(subspaces, axis=0)  # (50, r, 8)
        logger.info("  Rank %d: %d orthonormal subspaces (r x 8) verified.", r, n_per_rank)

    np.savez_compressed(bank_path, **bank_data)
    sha256_hash = compute_sha256(bank_path)

    manifest = {
        "status": "FROZEN_AND_HASHED",
        "dataset": "EchoNext",
        "description": "350 frozen subspaces in Grassmannian Gr(r, 8) for r in [1, 7]",
        "ranks": list(range(1, 8)),
        "n_subspaces_per_rank": n_per_rank,
        "total_subspaces": n_per_rank * 7,
        "ambient_dim": 8,
        "seed": seed,
        "sha256": sha256_hash,
        "file_path": str(bank_path),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Bank successfully created and hashed at %s", bank_path)
    logger.info("SHA-256: %s", sha256_hash)
    return bank_path


if __name__ == "__main__":
    generate_bank()
