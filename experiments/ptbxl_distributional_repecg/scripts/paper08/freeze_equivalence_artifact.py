#!/usr/bin/env python3
"""Mechanically audit and lock a completed Paper 08 development certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "manifest.json",
    "vocabulary_certificate.npz",
    "representation_train.npz",
    "representation_selection.npz",
    "representation_pseudotest.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit(artifact: Path) -> dict[str, object]:
    manifest = json.loads((artifact / "manifest.json").read_text())
    _require(manifest.get("kind") == "paper08_equivalence_token_representations", "wrong artifact kind")
    _require(manifest.get("status") == "complete", "certificate is not complete")
    _require(manifest.get("audit", {}).get("passed") is True, "development safeguards did not pass")
    _require(manifest.get("bootstrap") == {
        "unit": "patient", "replicates": 2000, "alpha": 0.05, "seed": 42,
    }, "bootstrap contract differs from the locked development specification")
    firewall = manifest.get("fold_firewall", {})
    _require(firewall.get("fold8_used_for_fit_or_selection") is False, "fold 8 was used before pseudo-test")
    _require(firewall.get("support_and_ucb_construction_folds") == [1, 2, 3, 4, 5, 6], "UCB folds differ")
    _require(firewall.get("delta_calibration_and_model_selection_fold") == 7, "delta/selection fold differs")
    _require(manifest.get("folds") == {
        "construction": [1, 2, 3, 4, 5, 6],
        "calibration_and_selection": [7],
        "pseudo_test": [8],
    }, "fold firewall declaration differs")
    _require(manifest.get("audit", {}).get("selection_unknown_rate", 1.0) <= 0.20, "UNK safeguard failed")
    _require(manifest.get("audit", {}).get("phase_nmi", 1.0) < 0.90, "phase-NMI safeguard failed")
    _require(np.isfinite(manifest.get("odd_even_stability", {}).get("same_record_odd_even", np.nan)), "missing odd/even stability")
    topology = manifest.get("phase_topology", {})
    _require(len(topology.get("phase_entropy_bits", [])) == manifest.get("final_vocabulary_size"), "phase topology missing")

    with np.load(artifact / "vocabulary_certificate.npz") as certificate:
        k0 = int(manifest["k0"])
        tokens = int(manifest["final_vocabulary_size"])
        _require(certificate["pair_bootstrap"].shape == (2000, k0 * (k0 - 1) // 2), "bootstrap draw shape differs")
        upper = certificate["simultaneous_upper"]
        _require(upper.shape == (k0, k0) and np.allclose(upper, upper.T), "invalid simultaneous UCB matrix")
        _require(np.isfinite(upper[np.triu_indices(k0, 1)]).all(), "nonfinite UCBs")
        base_to_token = certificate["base_to_token"]
        _require(np.all((base_to_token == -1) | ((0 <= base_to_token) & (base_to_token < tokens))), "invalid UNK/token map")
        _require(np.array_equal(np.flatnonzero(base_to_token < 0), certificate["unresolved_base_codes"]), "unresolved codes were assimilated")
        _require(certificate["random_merge_maps"].shape == (10, k0), "exact random controls missing")
        _require(certificate["frequency_matched_random_merge_maps"].shape == (10, k0), "frequency random controls missing")
        _require(certificate["token_phase_counts"].shape == (tokens + 1, 16), "phase counts differ")
        _require(np.isfinite(certificate["odd_even_different_patient_delta_draws"]).all(), "missing odd/even null")
        _require(np.isfinite(certificate["odd_even_random_token_delta_draws"]).all(), "missing random-token null")
        for merge in manifest["merge_tree"]:
            _require(merge["maximum_upper_bound"] < manifest["delta"]["q95"], "uncertified merge")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    lock_path = artifact / "artifact_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        current = {name: _sha256(artifact / name) for name in REQUIRED_FILES}
        _require(lock.get("sha256") == current, "existing artifact lock does not match artifact bytes")
        print(json.dumps(lock, sort_keys=True))
        return
    for name in REQUIRED_FILES:
        _require((artifact / name).is_file(), f"missing required file: {name}")
    manifest = _audit(artifact)
    lock = {
        "kind": "paper08_equivalence_artifact_lock",
        "certificate_status": manifest["status"],
        "sha256": {name: _sha256(artifact / name) for name in REQUIRED_FILES},
    }
    temporary = lock_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, lock_path)
    for name in (*REQUIRED_FILES, "artifact_lock.json"):
        (artifact / name).chmod(0o444)
    artifact.chmod(0o555)
    print(json.dumps(lock, sort_keys=True))


if __name__ == "__main__":
    main()
