#!/usr/bin/env python3
"""Generate, launch-gate, and operate the controlled ECG factorial benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.experiment_provenance import code_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(Path.home() / ".venv/bin/python")
RUN_ID = os.environ.get("FACTORIAL_RUN_ID", "factorial_v2")
RESULTS_ROOT = PROJECT_ROOT / "results" / RUN_ID
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / RUN_ID
DRIVER_ENV = f"FACTORIAL_RUN_ID={RUN_ID} PYTHONPATH=."
MASKS = [f"{value:04b}" for value in range(16)]
# cNVAE is intentionally retained in the implementation for later architecture
# work, but excluded from the active poster grid.  Its gate baseline converged
# to the four analytic limb leads while all five missing chest leads remained
# at approximately zero R² (2026-07-22; see results/factorial_v2/exclusions).
FAMILIES = ("unet", "msvae", "ecgaim")
SUPPORTED_FAMILIES = ("unet", "cnvae", "msvae", "ecgaim")
PRIMARY_SEED = 42
CONFIRMATION_SEEDS = (1337, 2026)
WEIGHTS = {"corr": 0.5, "mmd": 0.1, "deriv": 0.05}
OBSERVED = [0, 1, 7]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell_id(family: str, mask: str, seed: int) -> str:
    return f"{family}__e{mask[0]}c{mask[1]}m{mask[2]}d{mask[3]}__s{seed}"


def cell_dir(family: str, mask: str, seed: int, cohort: str = "primary") -> Path:
    return CHECKPOINT_ROOT / cohort / cell_id(family, mask, seed)


def checkpoint_path(family: str, mask: str, seed: int, cohort: str = "primary") -> Path:
    directory = cell_dir(family, mask, seed, cohort)
    return directory / ({"unet": "best.pt", "cnvae": "best_model.pt"}.get(family, "ul_ecp_best.pt"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_inventory(split: str) -> dict[str, Any]:
    root = PROJECT_ROOT / "data/ptb_xl/tensors" / split
    files = sorted(root.glob("*.pt"), key=lambda item: int(item.stem))
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.stem}:{path.stat().st_size}\n".encode())
    return {"count": len(files), "inventory_sha256": digest.hexdigest()}


def model_spec(family: str, mask: str, seed: int, cohort: str = "primary") -> dict[str, Any]:
    family_names = {
        "unet": ("unet", "unet"),
        "cnvae": ("cnvae", "cnvae"),
        "msvae": ("multiscale_vae", "msvae"),
        "ecgaim": ("ecg_aim", "alitok"),
    }
    display_family, kind = family_names[family]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    source_paths = {
        "unet": [
            "scripts/train_m1_pearson.py",
            "scripts/train_mcma_3lead.py",
            "scripts/common_loss.py",
        ],
        "cnvae": [
            "scripts/baselines/train_cnvae.py",
            "scripts/baselines/cnvae_ecg.py",
            "scripts/common_loss.py",
        ],
        "msvae": [
            "unified_latents/engineering/trainers/train_multi_scale_vae.py",
            "unified_latents/engineering/experimental/Multi_Scale_VAE.py",
            "scripts/common_loss.py",
        ],
        "ecgaim": [
            "unified_latents/engineering/trainers/train_multi_scale_vae.py",
            "unified_latents/engineering/experimental/alitok_vae_exp.py",
            "scripts/common_loss.py",
        ],
    }[family]
    architecture = {
        "unet": {"name": "MCMA-U-Net", "input_leads": [0, 1, 7]},
        "cnvae": {
            "name": "cNVAE-ECG-FM bridge",
            "backbone": "ECG-FM frozen",
            "inference_latent": "conditional_posterior_mean",
            "inference_inputs": "observed_leads_only",
            "numerical_precision": "float32_with_TF32",
        },
        "msvae": {"name": "WearECG MultiScale-VAE", "latent_channels": 4},
        "ecgaim": {"name": "ECG-AIM", "width": 384, "encoder_depth": 8, "decoder_depth": 4, "heads": 8, "patch_size": 25},
    }[family]
    optimizer = {
        "unet": {"name": "AdamW", "scheduler": None},
        "cnvae": {"name": "Adamax", "scheduler": "warmup+CosineAnnealingLR", "early_stopping_patience": 5},
        "msvae": {"name": "AdamW", "scheduler": "OneCycleLR"},
        "ecgaim": {"name": "AdamW", "scheduler": "OneCycleLR"},
    }[family]
    active = [
        name
        for name, bit in zip(("mse", "correlation", "mmd", "derivative"), mask)
        if bit == "1"
    ]
    return {
        "id": cell_id(family, mask, seed),
        "family": display_family,
        "kind": kind,
        "cohort": cohort,
        "seed": seed,
        "factorial_mask": mask,
        "factors": {"mse": mask[0] == "1", "correlation": mask[1] == "1", "mmd": mask[2] == "1", "derivative": mask[3] == "1"},
        "loss": "base" + ("+" + "+".join(active) if active else ""),
        "weights": WEIGHTS,
        "loss_implementation": "scripts.common_loss:{pearson_loss,mmd_loss,derivative_loss}",
        "mmd_implementation": "adaptive_multiscale_rbf_mean_squared_distance_v2",
        "base_objective": {"unet": "mse", "cnvae": "r2_elbo", "msvae": "vae_objective", "ecgaim": "tokenizer_reconstruction_objective"}[family],
        "checkpoint": str(checkpoint_path(family, mask, seed, cohort).relative_to(PROJECT_ROOT)),
        "observed_leads": OBSERVED,
        "architecture": architecture,
        "architecture_revision": revision,
        "architecture_provenance": code_provenance(PROJECT_ROOT, source_paths),
        "preprocessing": {"sample_rate_hz": 500, "shape": [12, 5000], "units": "mV", "normalization": "none", "split_inventory": {name: split_inventory(name) for name in ("train", "val", "test")}},
        "optimizer": optimizer,
        "checkpoint_selector": "lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson",
    }


def base_registry(models: list[dict[str, Any]], split: str = "test") -> dict[str, Any]:
    return {
        "schema_version": 3,
        "benchmark": RUN_ID,
        "data_dir": f"data/ptb_xl/tensors/{split}",
        "ptbxl_csv": "data/ptb_xl/ptbxl_database.csv",
        "scp_statements_csv": "data/ptb_xl/scp_statements.csv",
        "ecgfounder_repo": "ecg_fm_integration/ecgfounder_repo",
        "ecgfounder_checkpoint": "ecg_fm_integration/ecgfounder_repo/checkpoint/12_lead_ECGFounder.pth",
        "ecgfounder_labels_csv": "ecg_fm_integration/ecgfounder_repo/csv/ptbxl_label.csv",
        "ecgfounder_tasks": "ecg_fm_integration/ecgfounder_repo/tasks.txt",
        "five_superclass_backbone": "ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
        "five_superclass_checkpoint": str(
            (CHECKPOINT_ROOT / "parity/ecgfm_five_superclass.pt").relative_to(PROJECT_ROOT)
        ),
        "nstdb_dir": "data/mit-bih-noise-stress-test-database-1.0.0",
        "fitbit_noise_dir": "data/fitbit_noise",
        "split": split,
        "models": models,
    }


def train_command(family: str, mask: str, seed: int, *, smoke: bool = False, cohort: str = "primary", paper_parity: bool = False) -> str:
    output = cell_dir(family, mask, seed, cohort)
    epochs = 2 if smoke else (30 if family == "cnvae" else 10)
    weights = f"--lambda_corr {WEIGHTS['corr']} --lambda_mmd {WEIGHTS['mmd']} --lambda_deriv {WEIGHTS['deriv']}"
    common = f"--factorial_mask {mask}" if family == "unet" else (
        f"--lambda_mse {1.0 if mask[0] == '1' else 0.0} "
        f"--lambda_corr {WEIGHTS['corr'] if mask[1] == '1' else 0.0} "
        f"--lambda_mmd {WEIGHTS['mmd'] if mask[2] == '1' else 0.0} "
        f"--lambda_deriv {WEIGHTS['deriv'] if mask[3] == '1' else 0.0}"
    )
    if family == "unet":
        protocol = "paper_parity" if paper_parity else "extended"
        command = (
            f"PYTHONPATH=. {PYTHON} scripts/train_m1_pearson.py --seed {seed} --epochs {epochs} "
            f"--batch_size 256 --data_dir data/ptb_xl/tensors --loss_protocol {protocol} {common} {weights} "
            f"--run_name {cell_id(family, mask, seed)} --checkpoint_path {output / 'best.pt'} "
            f"--metadata_path {output / 'run_metadata.json'}"
        )
    elif family == "cnvae":
        smoke_arg = "--max_steps 20" if smoke else ""
        command = (
            f"PYTHONPATH=. {PYTHON} scripts/baselines/train_cnvae.py --backbone ecgfm "
            f"--ptbxl_root data/ptb_xl --tensor_dir data/ptb_xl/tensors --input_leads 0,1,7 --fm_checkpoint "
            f"ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt --epochs {epochs} "
            f"--patience 5 --beta 0.0001 --spectral_reg_weight 1e-6 --batchnorm_reg_weight 1e-6 --batch_size 16 --seed {seed} --save_dir {output.parent} --name {output.name} "
            f"--no-save_optimizer_state {common} {smoke_arg}"
        )
    else:
        model_family = "baseline" if family == "msvae" else "alitok"
        architecture = "" if family == "msvae" else (
            "--alitok_architecture ecg_aim_v1 --alitok_patch_size 25 --alitok_width 384 "
            "--alitok_encoder_depth 8 --alitok_decoder_depth 4 --alitok_heads 8 "
            "--alitok_lr 1e-4 --alitok_max_lr 5e-4"
        )
        smoke_arg = "--debug --fast_eval" if smoke else "--fast_eval"
        command = (
            f"PYTHONPATH=. {PYTHON} unified_latents/engineering/train_multi_scale_vae.py "
            f"--model_family {model_family} --data_dir data/ptb_xl/tensors --regime current "
            f"--no-fm_perceptual --no-mask_aware_encoder --no-split_latent --epochs {epochs} "
            f"--batch_size 32 --seed {seed} --save_dir {output} --run_tag {cell_id(family, mask, seed)} "
            f"{common} {architecture} {smoke_arg}"
        )
    return command


def phase_job(job_id: str, label: str, command: str, expected: Path | None = None) -> dict[str, Any]:
    job = {"id": job_id, "label": label, "cmd": command}
    if expected is not None:
        job["expected_output"] = str(expected.relative_to(PROJECT_ROOT))
    return job


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_models = [model_spec(family, mask, PRIMARY_SEED) for family in FAMILIES for mask in MASKS]
    paper_masks = ("1000", "1111", "1101", "1110", "1011")
    paper_models = [model_spec("unet", mask, PRIMARY_SEED, "paper_parity") for mask in paper_masks]
    registry = base_registry(primary_models, "test")
    validation_registry = base_registry(primary_models, "val")
    (output_dir / "model_registry.json").write_text(json.dumps(registry, indent=2))
    (output_dir / "validation_registry.json").write_text(json.dumps(validation_registry, indent=2))
    (output_dir / "paper_parity_registry.json").write_text(json.dumps(base_registry(paper_models, "test"), indent=2))
    (output_dir / "excluded_families.json").write_text(json.dumps({
        "cnvae": {
            "status": "excluded_after_validity_gate_diagnosis",
            "reason": "four analytic limb leads learned; five missing chest leads remained collapsed",
            "evidence": "results/factorial_v2/exclusions/cnvae.json",
        }
    }, indent=2))

    phases: list[dict[str, Any]] = []
    preflight_output = RESULTS_ROOT / "preflight.json"
    phases.append({"name": "preflight", "depends_on": [], "jobs": [phase_job(
        "factorial_preflight", "Storage, data, split, and dependency gates",
        f"{DRIVER_ENV} {PYTHON} scripts/factorial_v2.py preflight --output {preflight_output}", preflight_output)]})
    phases.append({"name": "loss_unit_tests", "depends_on": ["preflight"], "jobs": [phase_job(
        "factorial_loss_tests", "Loss routing, inference, robustness, statistics, and EchoNext unit tests",
        f"{DRIVER_ENV} {PYTHON} -m pytest -q "
        "tests/test_factorial_losses.py tests/test_factorial_analysis.py "
        "tests/test_robustness_stress.py tests/test_echonext_streaming.py")]})

    protocol_audit_output = RESULTS_ROOT / "protocol_audit.json"
    phases.append({"name": "protocol_audit", "depends_on": ["loss_unit_tests"], "jobs": [phase_job(
        "audit_generated_protocol", "Audit generated grid, commands, dependencies, and source hashes",
        f"{DRIVER_ENV} {PYTHON} scripts/factorial_v2.py audit-generated "
        f"--manifest {output_dir / 'manifest.json'} --registry {output_dir / 'model_registry.json'} "
        f"--output {protocol_audit_output}",
        protocol_audit_output)]})

    smoke_jobs = []
    # The original v2 queue already exercised the complete 48-cell routing
    # matrix. Corrective successor runs repeat the two configurations per
    # family that directly exercise the repaired MMD path at both MSE states;
    # the full-dimensional gradient regression test covers the kernel itself.
    smoke_masks = MASKS if RUN_ID == "factorial_v2" else ("0010", "1111")
    for family in FAMILIES:
        for mask in smoke_masks:
            expected = RESULTS_ROOT / "smoke" / f"{cell_id(family, mask, PRIMARY_SEED)}.json"
            smoke_jobs.append(phase_job(
                f"smoke_{cell_id(family, mask, PRIMARY_SEED)}", f"Smoke {family} {mask}",
                f"{DRIVER_ENV} {PYTHON} scripts/factorial_v2.py smoke-cell --family {family} --mask {mask} --seed {PRIMARY_SEED} --marker {expected}",
                expected))
    phases.append({"name": "smoke_factorial", "depends_on": ["protocol_audit"], "jobs": smoke_jobs})

    primary_jobs = []
    for family in FAMILIES:
        for mask in MASKS:
            expected = checkpoint_path(family, mask, PRIMARY_SEED)
            primary_jobs.append(phase_job(cell_id(family, mask, PRIMARY_SEED), f"Primary {family} {mask}", train_command(family, mask, PRIMARY_SEED), expected))
    phases.append({"name": "train_primary", "depends_on": ["smoke_factorial"], "jobs": primary_jobs})

    paper_jobs = [phase_job(
        f"paper_{cell_id('unet', mask, PRIMARY_SEED)}", f"Paper parity U-Net {mask}",
        train_command("unet", mask, PRIMARY_SEED, cohort="paper_parity", paper_parity=True),
        checkpoint_path("unet", mask, PRIMARY_SEED, "paper_parity")) for mask in paper_masks]
    phases.append({"name": "paper_parity_anchor", "depends_on": ["smoke_factorial"], "jobs": paper_jobs})

    validation_output = RESULTS_ROOT / "validation_results.json"
    phases.append({"name": "validation_evaluation", "depends_on": ["train_primary", "paper_parity_anchor"], "jobs": [
        phase_job("evaluate_primary_validation", "Evaluate all 48 active cells on validation only",
            f"{DRIVER_ENV} {PYTHON} scripts/evaluate_comprehensive_registry.py --registry {output_dir / 'validation_registry.json'} --output {validation_output} --skip_oracle --skip_robustness --batch_size 8 --morphology_samples 2183", validation_output),
    ]})
    phases.append({"name": "validation_selection", "depends_on": ["validation_evaluation"], "jobs": [
        phase_job("select_confirmation_masks", "Lock best nontrivial mask per family from validation",
            f"{DRIVER_ENV} {PYTHON} scripts/factorial_v2.py select --input {validation_output} --output {RESULTS_ROOT / 'selected_masks.json'}",
            RESULTS_ROOT / "selected_masks.json"),
    ]})

    confirmation_jobs = []
    for family in FAMILIES:
        for seed in CONFIRMATION_SEEDS:
            for slot in ("base", "full", "best"):
                marker = RESULTS_ROOT / f"confirmation/{family}__{slot}__s{seed}.json"
                confirmation_jobs.append(phase_job(
                    f"confirm_{family}_{slot}_s{seed}", f"Confirmation {family} {slot} seed {seed}",
                    f"{DRIVER_ENV} {PYTHON} scripts/factorial_v2.py train-selected --family {family} --slot {slot} --seed {seed} --selection {RESULTS_ROOT / 'selected_masks.json'} --marker {marker}", marker))
    phases.append({"name": "confirmation_seeds", "depends_on": ["validation_selection"], "jobs": confirmation_jobs})

    parity_checkpoint = CHECKPOINT_ROOT / "parity/ecgfm_five_superclass.pt"
    phases.append({"name": "five_superclass_parity", "depends_on": ["confirmation_seeds"], "jobs": [phase_job(
        "train_five_superclass_parity", "Retrain leakage-free ECG-FM five-superclass parity head",
        f"{DRIVER_ENV} {PYTHON} scripts/train_five_superclass_parity.py --output {parity_checkpoint} --epochs 20 --batch_size 32 --seed 42",
        parity_checkpoint)]})

    paper_test_output = RESULTS_ROOT / "paper_parity_results.json"
    phases.append({"name": "paper_parity_evaluation", "depends_on": ["five_superclass_parity", "paper_parity_anchor"], "jobs": [phase_job(
        "evaluate_paper_parity_anchor", "Evaluate five separately labeled paper-parity U-Net anchors",
        f"{DRIVER_ENV} {PYTHON} scripts/evaluate_comprehensive_registry.py --registry {output_dir / 'paper_parity_registry.json'} --output {paper_test_output} --batch_size 8 --skip_robustness --morphology_samples 2198",
        paper_test_output)]})

    clean_output = RESULTS_ROOT / "clean_results.json"
    phases.append({"name": "clean_ptbxl_evaluation", "depends_on": ["five_superclass_parity"], "jobs": [phase_job(
        "evaluate_factorial_clean_ptbxl", "All 48 active cells: clean PTB-XL, morphology, and diagnostic utility",
        f"{DRIVER_ENV} {PYTHON} scripts/evaluate_comprehensive_registry.py --registry {output_dir / 'model_registry.json'} --output {clean_output} --batch_size 8 --skip_robustness --morphology_samples 2198",
        clean_output)]})

    test_output = RESULTS_ROOT / "comprehensive_results.json"
    phases.append({"name": "full_robustness_evaluation", "depends_on": ["clean_ptbxl_evaluation", "paper_parity_evaluation"], "jobs": [phase_job(
        "evaluate_factorial_ptbxl", "All 48 active cells: clean, ECGFounder, morphology, and 17 stresses",
        f"{DRIVER_ENV} {PYTHON} scripts/evaluate_comprehensive_registry.py --registry {output_dir / 'model_registry.json'} --output {test_output} --batch_size 32 --robustness_samples 2198 --robustness_condition_batch 9 --morphology_samples 2198",
        test_output)]})

    stats_output = RESULTS_ROOT / "statistics.json"
    phases.append({"name": "factorial_statistics", "depends_on": ["full_robustness_evaluation"], "jobs": [phase_job(
        "analyze_factorial", "Paired marginal effects, interactions, CIs, and interim poster figures",
        f"{DRIVER_ENV} {PYTHON} scripts/analyze_factorial_v2.py --input {test_output} --registry {output_dir / 'model_registry.json'} --paper-parity {paper_test_output} --output-dir {RESULTS_ROOT}",
        stats_output)]})

    echonext_gate = RESULTS_ROOT / "echonext_preflight.json"
    echonext_results = RESULTS_ROOT / "echonext_results.json"
    phases.append({"name": "echonext_preflight", "depends_on": ["factorial_statistics"], "jobs": [
        phase_job("echonext_preflight", "EchoNext acquisition, units, lead-order, and provenance gate",
            f"{DRIVER_ENV} {PYTHON} scripts/factorial_v2.py echonext-preflight --output {echonext_gate}", echonext_gate),
    ]})
    phases.append({"name": "echonext_external", "depends_on": ["echonext_preflight"], "jobs": [
        phase_job("evaluate_echonext_factorial", "All 48 active primary cells on EchoNext",
            f"{DRIVER_ENV} {PYTHON} scripts/evaluate_echonext.py --registry {output_dir / 'model_registry.json'} --data_dir data/echonext --output {echonext_results} --batch_size 32 --robustness_condition_batch 9 --robustness_samples 2198 --morphology_samples 2198 --skip_ecgfounder --resume --device cuda", echonext_results),
    ]})

    smartwatch_results = RESULTS_ROOT / "smartwatch_results.json"
    phases.append({"name": "smartwatch_external", "depends_on": ["factorial_statistics"], "jobs": [
        phase_job("evaluate_smartwatch_factorial", "All active primary cells on Smartwatches zero-shot",
            f"{DRIVER_ENV} {PYTHON} scripts/evaluate_smartwatch.py --registry {output_dir / 'model_registry.json'} --data_dir data/physionet.org/files/ecg-capable-smartwatches/1.0.0 --output {smartwatch_results} --batch_size 16 --robustness_samples 0 --device cuda", smartwatch_results),
    ]})

    final_output = RESULTS_ROOT / "COMPLETENESS_REPORT.md"
    phases.append({"name": "poster_finalization", "depends_on": ["echonext_external", "smartwatch_external"], "jobs": [phase_job(
        "finalize_factorial_poster", "Completeness gate, EchoNext figure, tables, and final report",
        f"{DRIVER_ENV} {PYTHON} scripts/analyze_factorial_v2.py --input {test_output} --registry {output_dir / 'model_registry.json'} --paper-parity {paper_test_output} --echonext {echonext_results} --smartwatch {smartwatch_results} --output-dir {RESULTS_ROOT} --require-complete",
        final_output)]})

    manifest = {
        "project": f"ecg_{RUN_ID}",
        "cwd": str(PROJECT_ROOT),
        "ssh": "localhost",
        "conda": "none",
        "default_cmd": "echo no-default",
        "preconditions": [],
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 120, "max_attempts": 3},
        "phases": phases,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"registry_models": len(primary_models), "jobs": sum(len(p["jobs"]) for p in phases), "output_dir": str(output_dir)}, indent=2))


def preflight(output: Path, minimum_free_gib: float = 12.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    splits = {name: split_inventory(name) for name in ("train", "val", "test")}
    ids = {name: {int(path.stem) for path in (PROJECT_ROOT / f"data/ptb_xl/tensors/{name}").glob("*.pt")} for name in splits}
    overlaps = {"train_val": len(ids["train"] & ids["val"]), "train_test": len(ids["train"] & ids["test"]), "val_test": len(ids["val"] & ids["test"])}
    required = [
        "data/ptb_xl/ptbxl_database.csv",
        "data/ptb_xl/scp_statements.csv",
        "ecg_fm_integration/ecgfounder_repo/checkpoint/12_lead_ECGFounder.pth",
        "ecg_fm_integration/ecgfounder_repo/csv/ptbxl_label.csv",
        "ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
        "data/mit-bih-noise-stress-test-database-1.0.0/bw.dat",
        "data/mit-bih-noise-stress-test-database-1.0.0/em.dat",
        "data/mit-bih-noise-stress-test-database-1.0.0/ma.dat",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
    fitbit_provenance = PROJECT_ROOT / "data/fitbit_noise/PROVENANCE.json"
    payload = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "free_gib": free_gib,
        "minimum_free_gib": minimum_free_gib,
        "splits": splits,
        "overlaps": overlaps,
        "missing_required_assets": missing,
        "fitbit_noise_gate": {
            "provenance": str(fitbit_provenance.relative_to(PROJECT_ROOT)),
            "status": (
                "available_for_strict_validation"
                if fitbit_provenance.exists()
                else "not_included_missing_noise_extraction_provenance"
            ),
            "note": "The local smartwatch simulator files are not treated as a validated wearable-noise extraction.",
        },
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "status": "pass" if free_gib >= minimum_free_gib and not missing and not any(overlaps.values()) and splits["test"]["count"] == 2198 else "fail",
    }
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


def audit_generated(manifest_path: Path, registry_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    registry = json.loads(registry_path.read_text())
    expected_ids = {
        cell_id(family, mask, PRIMARY_SEED)
        for family in FAMILIES
        for mask in MASKS
    }
    models = registry.get("models", [])
    actual_ids = {model.get("id") for model in models}
    phases = {phase["name"]: phase for phase in manifest.get("phases", [])}
    jobs = [job for phase in manifest.get("phases", []) for job in phase.get("jobs", [])]
    job_ids = [job["id"] for job in jobs]
    source_hashes_valid = True
    model_contracts_valid = True
    for model in models:
        mask = model.get("factorial_mask", "")
        expected_factors = {
            "mse": len(mask) == 4 and mask[0] == "1",
            "correlation": len(mask) == 4 and mask[1] == "1",
            "mmd": len(mask) == 4 and mask[2] == "1",
            "derivative": len(mask) == 4 and mask[3] == "1",
        }
        model_contracts_valid &= model.get("seed") == PRIMARY_SEED
        model_contracts_valid &= model.get("factors") == expected_factors
        model_contracts_valid &= model.get("weights") == WEIGHTS
        model_contracts_valid &= model.get("mmd_implementation") == (
            "adaptive_multiscale_rbf_mean_squared_distance_v2"
        )
        model_contracts_valid &= model.get("checkpoint_selector") == (
            "lowest_missing_lead_val_mse_then_highest_missing_lead_val_pearson"
        )
        model_contracts_valid &= model.get("preprocessing", {}).get("normalization") == "none"
        provenance = model.get("architecture_provenance", {})
        for relative, expected_hash in provenance.get("source_file_sha256", {}).items():
            source_hashes_valid &= (PROJECT_ROOT / relative).is_file()
            source_hashes_valid &= sha256_file(PROJECT_ROOT / relative) == expected_hash

    primary_job_ids = {
        job["id"] for job in phases.get("train_primary", {}).get("jobs", [])
    }
    primary_commands = {
        job["id"]: job["cmd"]
        for job in phases.get("train_primary", {}).get("jobs", [])
    }
    throughput_contracts = True
    for job_id, command in primary_commands.items():
        family = job_id.split("__", 1)[0]
        expected_batch = {"unet": 256, "msvae": 32, "ecgaim": 32}[family]
        expected_epochs = 10
        throughput_contracts &= f"--batch_size {expected_batch}" in command
        throughput_contracts &= f"--epochs {expected_epochs}" in command

    confirmation_jobs = phases.get("confirmation_seeds", {}).get("jobs", [])
    expected_confirmation_runs = len(FAMILIES) * len(CONFIRMATION_SEEDS) * 3
    confirmation_contract = len(confirmation_jobs) == expected_confirmation_runs and all(
        "train-selected" in job["cmd"]
        and any(f"--seed {seed}" in job["cmd"] for seed in CONFIRMATION_SEEDS)
        for job in confirmation_jobs
    )
    split_inventories = {
        json.dumps(model["preprocessing"]["split_inventory"], sort_keys=True)
        for model in models
        if "preprocessing" in model
    }
    checks = {
        "registry_active_primary_cells_48": len(models) == 48 and actual_ids == expected_ids,
        "unique_job_ids": len(job_ids) == len(set(job_ids)),
        "primary_training_runs_48": primary_job_ids == expected_ids,
        "confirmation_runs_18": confirmation_contract,
        "paper_parity_anchor_runs_5": len(phases.get("paper_parity_anchor", {}).get("jobs", [])) == 5,
        "factor_and_selector_contracts": model_contracts_valid,
        "source_hashes_match_registry": source_hashes_valid,
        "identical_split_inventories": len(split_inventories) == 1,
        "throughput_and_epoch_contracts": throughput_contracts,
        "cnvae_excluded_from_active_registry": all(model.get("family") != "cnvae" for model in models),
        "active_grid_progresses_after_smoke": phases.get("train_primary", {}).get("depends_on") == ["smoke_factorial"],
        "full_robustness_all_records_and_safe_throughput": all(
            token in phases.get("full_robustness_evaluation", {}).get("jobs", [{}])[0].get("cmd", "")
            for token in (
                "--batch_size 32",
                "--robustness_samples 2198",
                "--robustness_condition_batch 9",
                "--morphology_samples 2198",
            )
        ),
        "external_datasets_required_for_finalization": set(
            phases.get("poster_finalization", {}).get("depends_on", [])
        ) == {"echonext_external", "smartwatch_external"},
        "echonext_full_external_contract": all(
            token in phases.get("echonext_external", {}).get("jobs", [{}])[0].get("cmd", "")
            for token in (
                "--batch_size 32",
                "--robustness_condition_batch 9",
                "--robustness_samples 2198",
                "--morphology_samples 2198",
            )
        ),
        "smartwatch_clean_only_contract": "--robustness_samples 0" in (
            phases.get("smartwatch_external", {}).get("jobs", [{}])[0].get("cmd", "")
        ),
    }
    payload = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "manifest": str(manifest_path),
        "registry": str(registry_path),
        "phase_count": len(phases),
        "job_count": len(jobs),
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, indent=2, allow_nan=False))
    if payload["status"] != "pass":
        raise SystemExit(6)


def cnvae_gate(input_path: Path, output: Path) -> None:
    results = json.loads(input_path.read_text())
    model = next(iter(results["models"].values()))
    primary = model["signal"]["missing_leads"]
    morphology = model["morphology"]
    checks = {
        "pearson_gte_0_70": primary["pearson"] >= 0.70,
        "r2_gte_0_50": primary["r2"] >= 0.50,
        "nonzero_peak_matches": morphology["n_peak_matches"] > 0,
    }
    payload = {"model": model["id"], "metrics": {"pearson": primary["pearson"], "r2": primary["r2"], "n_peak_matches": morphology["n_peak_matches"]}, "checks": checks, "status": "pass" if all(checks.values()) else "fail"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(3)


def select_masks(input_path: Path, output: Path) -> None:
    results = json.loads(input_path.read_text())["models"]
    selected: dict[str, str] = {}
    for family in FAMILIES:
        candidates = []
        for mask in MASKS:
            # Confirmation "best" belongs to the prespecified MSE-on 2^3
            # factorial. MSE-off cells are supplemental and cannot win this
            # validation selection.
            if mask[0] != "1" or mask in {"1000", "1111"}:
                continue
            model = results[cell_id(family, mask, PRIMARY_SEED)]
            metric = model["signal"]["missing_leads"]
            candidates.append((metric["mse"], -metric["pearson"], mask))
        selected[family] = min(candidates)[2]
    payload = {
        "selected_on": "validation",
        "candidate_space": "mse_on_nontrivial_2^3",
        "selector": "lowest_missing_lead_mse_then_highest_pearson",
        "mask_encoding": "ecmd_four_bit",
        "masks": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def train_selected(family: str, slot: str, seed: int, selection: Path, marker: Path) -> None:
    mask = {"base": "1000", "full": "1111"}.get(slot)
    if mask is None:
        mask = json.loads(selection.read_text())["masks"][family]
    command = train_command(family, mask, seed, cohort="confirmation")
    subprocess.run(command, shell=True, cwd=PROJECT_ROOT, check=True, executable="/bin/bash")
    checkpoint = checkpoint_path(family, mask, seed, "confirmation")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    spec = model_spec(family, mask, seed, "confirmation")
    registry_path = marker.with_suffix(".registry.json")
    evaluation_path = marker.with_suffix(".evaluation.json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(base_registry([spec], "val"), indent=2))
    subprocess.run([
        PYTHON,
        "scripts/evaluate_comprehensive_registry.py",
        "--registry", str(registry_path),
        "--output", str(evaluation_path),
        "--batch_size", "8",
        "--skip_oracle",
        "--skip_robustness",
        "--morphology_samples", "2183",
    ], cwd=PROJECT_ROOT, check=True)
    evaluated = json.loads(evaluation_path.read_text())["models"][spec["id"]]
    signal = evaluated["signal"]["missing_leads"]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "family": family,
        "slot": slot,
        "mask": mask,
        "seed": seed,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(checkpoint),
        "validation_evaluation": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "validation_metrics": {
            "n_samples": evaluated["signal"]["n_samples"],
            "mse": signal["mse"],
            "r2": signal["r2"],
            "pearson": signal["pearson"],
        },
    }, indent=2))


def smoke_cell(family: str, mask: str, seed: int, marker: Path) -> None:
    """Train and adapter-load one short cell, then remove temporary weights."""
    command = train_command(family, mask, seed, smoke=True, cohort="smoke")
    subprocess.run(command, shell=True, cwd=PROJECT_ROOT, check=True, executable="/bin/bash")
    checkpoint = checkpoint_path(family, mask, seed, "smoke")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    spec = model_spec(family, mask, seed, "smoke")
    registry_path = marker.with_suffix(".registry.json")
    evaluation_path = marker.with_suffix(".evaluation.json")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(base_registry([spec], "val"), indent=2))
    evaluation_command = [
        PYTHON,
        "scripts/evaluate_comprehensive_registry.py",
        "--registry", str(registry_path),
        "--output", str(evaluation_path),
        "--max_samples", "1",
        "--batch_size", "1",
        "--num_workers", "0",
        "--skip_oracle",
        "--skip_robustness",
        "--morphology_samples", "1",
    ]
    subprocess.run(evaluation_command, cwd=PROJECT_ROOT, check=True)
    evaluated = json.loads(evaluation_path.read_text())["models"][spec["id"]]
    observed_mse = evaluated["signal"]["observed_leads"]["mse"]
    if observed_mse != 0.0:
        raise RuntimeError(f"Observed-lead preservation failed: mse={observed_mse}")
    marker.write_text(json.dumps({
        "id": spec["id"],
        "status": "pass",
        "checkpoint_sha256": sha256_file(checkpoint),
        "output_shape": [1, 12, 5000],
        "finite": True,
        "observed_lead_mse": observed_mse,
    }, indent=2))
    registry_path.unlink(missing_ok=True)
    evaluation_path.unlink(missing_ok=True)
    shutil.rmtree(cell_dir(family, mask, seed, "smoke"))


def echonext_preflight(output: Path) -> None:
    root = PROJECT_ROOT / "data/echonext"
    waveform = next(iter(root.glob("**/EchoNext_test_waveforms.npy")), None) if root.exists() else None
    provenance = root / "PROVENANCE.json"
    payload = {"data_root": str(root), "waveform": str(waveform) if waveform else None, "provenance": str(provenance), "status": "pass" if waveform and provenance.exists() else "blocked_missing_data_or_provenance"}
    if payload["status"] == "pass":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(4)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment_queue" / RUN_ID,
    )
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--output", type=Path, required=True)
    audit_parser = sub.add_parser("audit-generated")
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.add_argument("--registry", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    gate_parser = sub.add_parser("cnvae-gate")
    gate_parser.add_argument("--input", type=Path, required=True)
    gate_parser.add_argument("--output", type=Path, required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--input", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    selected_parser = sub.add_parser("train-selected")
    selected_parser.add_argument("--family", choices=FAMILIES, required=True)
    selected_parser.add_argument("--slot", choices=("base", "full", "best"), required=True)
    selected_parser.add_argument("--seed", type=int, required=True)
    selected_parser.add_argument("--selection", type=Path, required=True)
    selected_parser.add_argument("--marker", type=Path, required=True)
    smoke_parser = sub.add_parser("smoke-cell")
    smoke_parser.add_argument("--family", choices=FAMILIES, required=True)
    smoke_parser.add_argument("--mask", choices=MASKS, required=True)
    smoke_parser.add_argument("--seed", type=int, required=True)
    smoke_parser.add_argument("--marker", type=Path, required=True)
    echo_parser = sub.add_parser("echonext-preflight")
    echo_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate": generate(args.output_dir)
    elif args.command == "preflight": preflight(args.output)
    elif args.command == "audit-generated": audit_generated(args.manifest, args.registry, args.output)
    elif args.command == "cnvae-gate": cnvae_gate(args.input, args.output)
    elif args.command == "select": select_masks(args.input, args.output)
    elif args.command == "train-selected": train_selected(args.family, args.slot, args.seed, args.selection, args.marker)
    elif args.command == "smoke-cell": smoke_cell(args.family, args.mask, args.seed, args.marker)
    elif args.command == "echonext-preflight": echonext_preflight(args.output)


if __name__ == "__main__":
    main()
