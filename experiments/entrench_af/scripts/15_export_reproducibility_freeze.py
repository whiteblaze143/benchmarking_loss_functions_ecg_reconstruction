#!/usr/bin/env python3
"""
Stage 15 — Final Reproducibility Freeze Exporter.

Creates an immutable archival freeze bundle under:
  results/reproducibility_freeze_<run_name>/
containing:
  - exact modified source tree (`scripts/`, `config/`, `tests/`)
  - `working_tree.diff` & `git_status.txt`
  - `environment_pip_freeze.txt` (Python, OS, full installed packages)
  - `seeds_and_hashes.json` (fixed RNG seeds across all stages, definitions hash, git hash)
  - `input_dataset_checksums.sha256` & `dataset_versions.json`
  - final deliverables (`HRS_2027_Abstract.md`, `CLAIM_REGISTRY.md`, `ENTRENCH_AF_Report.md`,
    `HRS_Principal_Figure.png`/`.pdf`, consolidated `tables/`, bootstrap replicates,
    identifiability audit, coverage sensitivity, run ledger, delayed-entry verification,
    and pytest verification log).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import common  # type: ignore


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export immutable reproducibility freeze package.")
    parser.add_argument("--pytest-log", type=str, default=None, help="Path to captured pytest verification log.")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir()
    freeze_dir = PROJECT_ROOT / "results" / f"reproducibility_freeze_{run_dir.name}"
    freeze_dir.mkdir(parents=True, exist_ok=True)

    # 1. Archive exact source tree (`scripts/`, `config/`, `tests/`)
    src_dir = freeze_dir / "source"
    for sub in ("scripts", "config", "tests"):
        s_path = PROJECT_ROOT / sub
        if s_path.exists():
            shutil.copytree(
                s_path,
                src_dir / sub,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )

    # 2. Capture git working-tree diff & status
    repo_root = PROJECT_ROOT.parents[1]
    diff_txt = subprocess.check_output(["git", "diff", "HEAD", "--", "experiments/entrench_af"], cwd=str(repo_root), text=True, errors="replace")
    (freeze_dir / "working_tree.diff").write_text(diff_txt, encoding="utf-8")
    status_txt = subprocess.check_output(["git", "status", "--short", "--", "experiments/entrench_af"], cwd=str(repo_root), text=True, errors="replace")
    (freeze_dir / "git_status.txt").write_text(status_txt, encoding="utf-8")

    # 3. Capture dependency environment & runtime info
    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, errors="replace")
    env_header = (
        f"# Python: {sys.version.replace(chr(10), ' ')}\n"
        f"# Platform: {platform.platform()}\n"
        f"# TimestampUTC: {common.utc_timestamp_iso()}\n"
    )
    (freeze_dir / "environment_pip_freeze.txt").write_text(env_header + pip_freeze, encoding="utf-8")

    # 4. Seeds, definitions hash, & git hash
    seeds_meta = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "git_hash": common.get_git_hash(),
        "definitions_hash": common.compute_definitions_hash(),
        "exported_utc": common.utc_timestamp_iso(),
        "random_seeds": {
            "global_seed": 20260401,
            "aim1_bootstrap_km_and_na_seed": 20260401,
            "aim1_delayed_entry_verification_seed": 20260401 + 31,
            "aim2_threshold_bootstrap_seed": 20260401,
            "aim2_paired_bootstrap_seed": 20260401 + 99,
            "secondary_icc_bootstrap_seed": 20260401,
        },
        "source_sha256": {
            str(p.relative_to(freeze_dir)): _sha256_file(p)
            for p in sorted(src_dir.rglob("*"))
            if p.is_file()
        },
    }
    (freeze_dir / "seeds_and_hashes.json").write_text(json.dumps(seeds_meta, indent=2), encoding="utf-8")

    # 5. Input dataset versions & metadata/config checksums
    cohorts_cfg = yaml.safe_load((PROJECT_ROOT / "config" / "cohorts.yaml").read_text(encoding="utf-8"))
    ds_versions = {}
    checksum_lines = []
    for cname, cinfo in cohorts_cfg.get("cohorts", {}).items():
        ds_info = {"config": cinfo}
        for key in ("metadata_csv", "records_file"):
            if key in cinfo and Path(cinfo[key]).exists():
                p = Path(cinfo[key])
                sha = _sha256_file(p)
                ds_info[f"{key}_sha256"] = sha
                checksum_lines.append(f"{sha}  {p}")
        data_root = Path(cinfo.get("raw_dir", cinfo.get("root_dir", "")))
        if data_root.exists():
            ds_info["raw_dir_exists"] = True
            ds_info["raw_file_count"] = sum(1 for _ in data_root.rglob("*") if _.is_file())
        ds_versions[cname] = ds_info
    for cfg_p in sorted((PROJECT_ROOT / "config").glob("*.yaml")):
        checksum_lines.append(f"{_sha256_file(cfg_p)}  {cfg_p}")
    for p_inv in [run_dir / "stage0" / "record_inventory.csv", run_dir / "stage1" / "beat_summary.parquet"]:
        if p_inv.exists():
            checksum_lines.append(f"{_sha256_file(p_inv)}  {p_inv}")
    (freeze_dir / "dataset_versions.json").write_text(json.dumps(ds_versions, indent=2), encoding="utf-8")
    (freeze_dir / "input_dataset_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # 6. Export final deliverables, patient/episode result tables, and bootstrap replicates
    deliv_dir = freeze_dir / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "reports/HRS_2027_Abstract.md",
        "reports/CLAIM_REGISTRY.md",
        "reports/ENTRENCH_AF_Report.md",
        "reports/figures/HRS_Principal_Figure.png",
        "reports/figures/HRS_Principal_Figure.pdf",
    ]:
        sp = PROJECT_ROOT / rel
        if sp.exists():
            shutil.copy2(sp, deliv_dir / sp.name)

    tab_src = PROJECT_ROOT / "reports" / "tables"
    if tab_src.exists():
        shutil.copytree(tab_src, deliv_dir / "tables", dirs_exist_ok=True)

    run_artifacts = [
        run_dir / "stage4" / "events_g0.parquet",
        run_dir / "aim2" / "reconstructed_episodes.parquet",
        run_dir / "aim2" / "paired_bootstrap_replicates.csv",
        run_dir / "aim2" / "bootstrap_identifiability_audit.csv",
        run_dir / "aim2" / "coverage_sensitivity_aggregation.csv",
        run_dir / "aim2" / "coverage_sensitivity_threshold.csv",
        run_dir / "aim2" / "matched_eligibility_run_ledger.csv",
        run_dir / "aim1" / "delayed_entry_likelihood_verification.csv",
    ]
    art_out = deliv_dir / "artifacts"
    art_out.mkdir(parents=True, exist_ok=True)
    for ap in run_artifacts:
        if ap.exists():
            shutil.copy2(ap, art_out / ap.name)

    if args.pytest_log and Path(args.pytest_log).exists():
        shutil.copy2(args.pytest_log, deliv_dir / "pytest_verification.log")

    print(f"[Stage 15] Reproducibility freeze exported cleanly -> {freeze_dir}")


if __name__ == "__main__":
    main()
