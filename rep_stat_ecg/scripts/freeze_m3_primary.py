"""Freeze completed adapted-repSpat M3 and record the failed quotient gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path("refine-logs/qvcg")
FROZEN = ROOT / "m3_adapted_repspat_primary_frozen"
FILES = [
    "VCG_REPSPAT_PAIR_TESTS.parquet",
    "VCG_MMD_MATRIX.npy",
    "VCG_MOTIF_REGISTRY.parquet",
    "VCG_VOXEL_TO_MOTIF.parquet",
    "VCG_MOTIF_SUMMARY.json",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source, destination = ROOT / name, FROZEN / name
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() and digest(destination) != digest(source):
            raise RuntimeError(f"Refusing to replace non-identical frozen artifact: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)

    pairs = pd.read_parquet(FROZEN / "VCG_REPSPAT_PAIR_TESTS.parquet")
    motifs = pd.read_parquet(FROZEN / "VCG_MOTIF_REGISTRY.parquet")
    if len(pairs) != 2016 or int(pairs.bh_reject_imq.sum()) != 1063:
        raise RuntimeError("Primary M3 pair invariants do not match the completed run")
    if not (
        len(motifs) == 1 and int(motifs.iloc[0].num_domains) == 64
        and motifs.iloc[0].status == "AMBIGUOUS_NON_CLIQUE"
        and not bool(motifs.iloc[0].is_clique)
    ):
        raise RuntimeError("Primary M3 quotient invariant does not match failed gate")

    manifest = {
        "run_id": "M3_ADAPTED_REPSPAT_PRIMARY",
        "status": "COMPLETE_FROZEN",
        "graph_result": "SINGLE_AMBIGUOUS_NON_CLIQUE_COMPONENT",
        "quotient_status": "FAIL",
        "pairwise_structure_status": "NONTRIVIAL",
        "n_domains": 64, "n_pairs": 2016,
        "n_bh_rejected": 1063, "n_bh_nonrejected": 953,
        "sha256": {name: digest(FROZEN / name) for name in FILES},
        "source_paths": {name: str(ROOT / name) for name in FILES},
    }
    gate = {
        "gate": "QVCG_CC_MOTIF_GATE", "status": "FAIL",
        "reason": "All 64 domains form one AMBIGUOUS_NON_CLIQUE connected component despite 1,063 directly rejected pairs.",
        "component_is_valid_motif": False,
        "QVCG_RECURRENT_STRUCTURE": "UNRESOLVED",
        "prohibited_interpretation": "The single connected component is not a valid motif or equivalence class.",
    }
    (FROZEN / "M3_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "QVCG_CC_MOTIF_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    for name in FILES:
        (FROZEN / name).chmod(0o444)
    print(json.dumps({"manifest": manifest, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
