import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.figures.clinical_common import selected_ids
from scripts.factorial_v2 import FAMILIES, MASKS, cell_id, select_masks
from scripts.check_factorial_v4_final import (
    CLINICAL_FIGURES,
    CORE_FIGURES,
    queue_complete,
    review_complete,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reuse_manifest_binds_corrected_registry_and_external_results(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_factorial_v3_queue.py",
            "--output",
            str(output),
            "--registry",
            "experiment_queue/factorial_v4/model_registry.json",
            "--source-results",
            "results/factorial_v4",
            "--evaluation-results",
            "results/factorial_v4",
            "--result-dir",
            "results/factorial_v4_clinical",
            "--project",
            "ecg_factorial_v4_clinical",
            "--reuse-external",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    manifest = json.loads(output.read_text())
    phases = {phase["name"]: phase for phase in manifest["phases"]}
    jobs = {
        job["id"]: job
        for phase in manifest["phases"]
        for job in phase["jobs"]
    }

    assert manifest["project"] == "ecg_factorial_v4_clinical"
    assert "echonext_shd" not in phases
    assert "smartwatch" not in phases
    assert phases["clinical_databases"]["depends_on"] == ["ptbxl_task_backfill"]
    assert (
        "--comprehensive results/factorial_v4/comprehensive_results.json"
        in jobs["v3_ptbxl_task_backfill"]["cmd"]
    )
    assert (
        "--comprehensive results/factorial_v4/comprehensive_results.json"
        in jobs["v3_preflight"]["cmd"]
    )
    database_cmd = jobs["v3_clinical_databases"]["cmd"]
    assert "--echonext results/factorial_v4/echonext_results.json" in database_cmd
    assert "--smartwatch results/factorial_v4/smartwatch_results.json" in database_cmd
    assert "--registry experiment_queue/factorial_v4/model_registry.json" in database_cmd
    assert (
        "--evaluation_dir results/factorial_v4"
        in jobs["v3_completeness"]["cmd"]
    )
    for identifier in (
        "fig_ptbxl_task_rank",
        "fig_ptbxl_superclasses",
        "fig_echonext_shd",
        "fig_echonext_stress",
        "fig_smartwatch_radar",
        "fig_smartwatch_task_rank",
        "fig_echonext_calibration",
    ):
        assert (
            "--selection results/factorial_v4/selected_masks.json"
            in jobs[identifier]["cmd"]
        )


def test_selected_ids_support_legacy_and_corrected_mask_encodings(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"masks": {"unet": "101", "msvae": "110", "ecgaim": "100"}}))
    corrected = tmp_path / "corrected.json"
    corrected.write_text(json.dumps({"masks": {"unet": "1101", "msvae": "1110", "ecgaim": "1100"}}))

    assert selected_ids(legacy)["unet"] == "unet__e1c1m0d1__s42"
    assert selected_ids(corrected)["unet"] == "unet__e1c1m0d1__s42"


def test_selected_ids_reject_mse_off_corrected_selection(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"masks": {"unet": "0101", "msvae": "1110", "ecgaim": "1100"}}))
    with pytest.raises(ValueError, match="must be MSE-on"):
        selected_ids(invalid)


def test_validation_selection_is_restricted_to_mse_on_factorial(tmp_path):
    models = {}
    for family in FAMILIES:
        for mask in MASKS:
            if mask[0] != "1" or mask in {"1000", "1111"}:
                continue
            mse = 0.01 if mask == "1010" else 0.1
            models[cell_id(family, mask, 42)] = {
                "signal": {"missing_leads": {"mse": mse, "pearson": 0.8}}
            }
    source = tmp_path / "validation.json"
    output = tmp_path / "selected.json"
    source.write_text(json.dumps({"models": models}))

    select_masks(source, output)
    selected = json.loads(output.read_text())

    assert selected["candidate_space"] == "mse_on_nontrivial_2^3"
    assert selected["mask_encoding"] == "ecmd_four_bit"
    assert selected["masks"] == {family: "1010" for family in FAMILIES}


def test_final_gate_helpers_are_fail_closed(tmp_path):
    assert queue_complete({"jobs": [{"status": "completed"}]})
    assert not queue_complete({"jobs": []})
    assert not queue_complete({"jobs": [{"status": "running"}]})
    core_dir = tmp_path / "core"
    clinical_dir = tmp_path / "clinical"
    core_dir.mkdir()
    clinical_dir.mkdir()
    core_review = {
        "figures": {
            name: {
                "data_hash_gate": "PASS",
                "visual_review": "PASS",
                "png_sha256": "",
            }
            for name in CORE_FIGURES
        }
    }
    clinical_review = {
        "figures": {
            name: {
                "data_hash_gate": "PASS",
                "visual_review": "PASS",
                "png_sha256": "",
            }
            for name in CLINICAL_FIGURES
        }
    }
    for name in CORE_FIGURES:
        path = core_dir / f"{name}.png"
        path.write_bytes(name.encode())
        core_review["figures"][name]["png_sha256"] = sha256(path)
    for name in CLINICAL_FIGURES:
        path = clinical_dir / f"{name}.png"
        path.write_bytes(name.encode())
        clinical_review["figures"][name]["png_sha256"] = sha256(path)
    assert review_complete(core_review, CORE_FIGURES, core_dir)
    assert review_complete(clinical_review, CLINICAL_FIGURES, clinical_dir)
    clinical_review["figures"].pop(next(iter(CLINICAL_FIGURES)))
    assert not review_complete(clinical_review, CLINICAL_FIGURES, clinical_dir)
