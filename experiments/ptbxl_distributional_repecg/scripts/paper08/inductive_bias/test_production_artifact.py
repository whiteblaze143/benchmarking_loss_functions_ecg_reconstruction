from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARTIFACT = Path(
    "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
    "paper08_tokens/development_representations"
)


def test_certificate_complete_manifest_and_fallback_gate() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    assert manifest["kind"] == "paper08_equivalence_token_representations"
    assert manifest["status"] == "complete"
    assert manifest["audit"]["passed"] is True
    assert [item["k0"] for item in manifest["k0_support_audit"]] == [512, 256]
    assert manifest["k0_support_audit"][0]["unsupported_fraction"] > 0.20
    assert manifest["k0_support_audit"][1]["unsupported_fraction"] <= 0.20
    assert manifest["supported_base_codes"] == 229
    assert manifest["certified_merges"] == len(manifest["merge_tree"]) == 113
    assert manifest["final_vocabulary_size"] == 116
    assert manifest["supported_base_codes"] - manifest["certified_merges"] == 116


def test_certificate_controls_and_shapes_reconcile() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    with np.load(ARTIFACT / "vocabulary_certificate.npz") as item:
        assert item["simultaneous_upper"].shape == (256, 256)
        assert item["pair_bootstrap"].shape == (100, 256 * 255 // 2)
        assert item["random_merge_maps"].shape == (10, 256)
        assert item["frequency_matched_random_merge_maps"].shape == (10, 256)
        assert item["token_phase_counts"].shape == (117, 16)
        base_to_token = item["base_to_token"]
        sizes = sorted(int(np.sum(base_to_token == token)) for token in range(116))
        for key in ("random_merge_maps", "frequency_matched_random_merge_maps"):
            for mapping in item[key]:
                assert sorted(int(np.sum(mapping == token)) for token in range(116)) == sizes
    assert manifest["audit"]["validation_unknown_rate"] <= 0.20
    assert manifest["audit"]["phase_nmi"] < 0.90
    assert len(manifest["validation_unknown_rate_by_phase"]) == 16
    assert set(manifest["validation_unknown_rate_by_diagnosis"]) == {
        "NORM", "MI", "STTC", "CD", "HYP",
    }


def test_representations_are_finite_and_token_ranges_are_valid() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    for split, records in (("train", 15_244), ("val", 2_173)):
        with np.load(ARTIFACT / f"representation_{split}.npz") as item:
            assert item["continuous"].shape == (records, 16, 256)
            assert item["equivalence_token_ids"].shape == (records, 16)
            assert np.isfinite(item["continuous"]).all()
            assert item["equivalence_token_ids"].min() >= -1
            assert item["equivalence_token_ids"].max() < manifest["final_vocabulary_size"]
