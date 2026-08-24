#!/usr/bin/env python3
"""Verify the EMBC poster's evidence, PDF geometry, and print-facing contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
PDF = OUT / "poster.pdf"
TEX = OUT / "poster.tex"
LOG = OUT / "build" / "poster.log"
MANIFEST = OUT / "ASSET_MANIFEST.json"
POPPLER_BIN = ROOT / "poster_html" / "tools" / "usr" / "bin"
POPPLER_LIB = ROOT / "poster_html" / "tools" / "usr" / "lib" / "x86_64-linux-gnu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def poppler(command: str, *args: str) -> str:
    env = os.environ.copy()
    env["PATH"] = f"{POPPLER_BIN}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{POPPLER_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(
        [command, *args],
        cwd=OUT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    record("poster_pdf_exists", PDF.is_file(), str(PDF))
    record("poster_tex_exists", TEX.is_file(), str(TEX))

    info = poppler("pdfinfo", str(PDF))
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    size_match = re.search(
        r"^Page size:\s+([\d.]+) x ([\d.]+) pts", info, re.MULTILINE
    )
    width_pt, height_pt = map(float, size_match.groups())
    width_mm = width_pt / 72 * 25.4
    height_mm = height_pt / 72 * 25.4
    record("single_page", pages == 1, f"{pages} page")
    record(
        "a0_portrait_geometry",
        abs(width_mm - 841) < 1.0 and abs(height_mm - 1189) < 1.0,
        f"{width_mm:.2f} x {height_mm:.2f} mm",
    )

    extracted = poppler("pdftotext", "-layout", str(PDF), "-")
    required_phrases = [
        "Benchmarking Loss Functions for 12-Lead",
        "ECG Reconstruction from Limited Leads",
        "Mithun Manivannan",
        "Experimental Design",
        "Complete 2",
        "primary factorial models",
        "Declaration of interest",
    ]
    normalized_text = re.sub(r"\s+", " ", extracted)
    for phrase in required_phrases:
        record(
            f"text_present:{phrase}",
            phrase in normalized_text,
            "found" if phrase in normalized_text else "missing",
        )

    log_text = LOG.read_text(errors="replace")
    overfull_vboxes = re.findall(r"Overfull \\\\vbox", log_text)
    overfull_hboxes = re.findall(r"Overfull \\\\hbox", log_text)
    record("no_overfull_vboxes", not overfull_vboxes, f"{len(overfull_vboxes)}")
    # Long URLs can generate a harmless sub-point hbox warning; report rather than hide it.
    record(
        "overfull_hbox_count_below_3",
        len(overfull_hboxes) < 3,
        f"{len(overfull_hboxes)}",
    )

    manifest = json.loads(MANIFEST.read_text())
    mismatches = []
    for item in manifest["assets"]:
        path = OUT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            mismatches.append(item["path"])
    record("asset_manifest_hashes", not mismatches, ", ".join(mismatches) or "all match")

    master = pd.read_csv(OUT / "data" / "all_48_models_master.csv")
    record("master_has_48_cells", len(master) == 48, str(len(master)))
    design = master.groupby("family").size().to_dict()
    record(
        "factorial_design_is_3x16",
        sorted(design.values()) == [16, 16, 16],
        json.dumps(design, sort_keys=True),
    )
    complete = json.loads((OUT / "data" / "completeness_verification.json").read_text())
    complete_pass = (
        complete.get("overall_status") == "pass"
        or complete.get("status") in {"pass", "complete"}
        or all(item.get("passed") for item in complete.get("checks", []))
    )
    record("upstream_completeness_passes", complete_pass, str(complete.get("status")))

    tex_text = TEX.read_text()
    exact_claims = {
        "MSVAE correlation-only Pearson": "0.874",
        "MSVAE correlation-only R2": "-14.984",
        "U-Net best R2": "0.700",
        "MSVAE best R2": "0.816",
        "ECG-AIM best R2": "0.822",
        "ECG-AIM full Sunnybrook Pearson": "0.849",
    }
    for name, value in exact_claims.items():
        record(f"claim_literal:{name}", value in tex_text, value)

    unresolved_declaration = (
        "author confirmation required" in tex_text.lower()
        or "Author confirmation required" in tex_text
    )
    record(
        "author_declaration_flagged",
        unresolved_declaration,
        "Funding/conflict macros intentionally require author confirmation",
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "schema": 1,
        "status": "pass_with_author_declaration_pending" if passed else "fail",
        "poster": str(PDF.relative_to(ROOT)),
        "geometry_mm": [round(width_mm, 2), round(height_mm, 2)],
        "checks_passed": sum(bool(item["passed"]) for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "author_action_required": (
            "Replace FundingStatement and ConflictStatement macros before printing."
        ),
    }
    (OUT / "VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")

    deliverables = [
        OUT / "poster.tex",
        OUT / "poster.pdf",
        OUT / "poster_preview.png",
        OUT / "ASSET_MANIFEST.json",
        OUT / "VERIFICATION.json",
    ]
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in deliverables)
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
