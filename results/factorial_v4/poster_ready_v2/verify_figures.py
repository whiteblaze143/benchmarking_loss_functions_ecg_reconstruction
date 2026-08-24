#!/usr/bin/env python3
"""Fail-closed integrity and coverage checks for the poster-ready figure set."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGURES = HERE / "figures"
EXPECTED = [
    "fig1_loss_landscape",
    "fig2_component_effects",
    "fig3_morphology_utility",
    "fig4_robustness_external",
    "fig5_smartwatch_domain_gap",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "pass": bool(condition)})


def main() -> None:
    checks: list[dict] = []
    for stem in EXPECTED:
        provenance_path = FIGURES / f"{stem}.provenance.json"
        check(provenance_path.exists(), f"{stem}: provenance exists", checks)
        if not provenance_path.exists():
            continue
        provenance = json.loads(provenance_path.read_text())
        check(
            bool(provenance.get("caption")),
            f"{stem}: nonempty caption",
            checks,
        )
        for source_text, expected_hash in provenance.get("sources", {}).items():
            source = Path(source_text)
            check(source.exists(), f"{stem}: source exists: {source.name}", checks)
            if source.exists():
                check(
                    sha256(source) == expected_hash,
                    f"{stem}: source hash matches: {source.name}",
                    checks,
                )
        for suffix in ("pdf", "svg", "png"):
            output = FIGURES / f"{stem}.{suffix}"
            recorded = provenance.get("outputs", {}).get(suffix, {})
            check(output.exists(), f"{stem}: {suffix} exists", checks)
            if output.exists():
                check(output.stat().st_size > 1_000, f"{stem}: {suffix} nonempty", checks)
                check(
                    sha256(output) == recorded.get("sha256"),
                    f"{stem}: {suffix} hash matches",
                    checks,
                )

    primary = pd.read_parquet(
        ROOT / "results/factorial_v4_2x4/main_48_cell_table.parquet"
    )
    check(len(primary) == 48, "primary table has 48 rows", checks)
    check(
        primary[["family", "mask"]].drop_duplicates().shape[0] == 48,
        "primary table has 48 unique family-mask cells",
        checks,
    )
    check(
        set(primary["family"]) == {"unet", "msvae", "ecgaim"},
        "primary table contains exactly three valid families",
        checks,
    )
    check(
        set(primary["mask"].astype(str).str.zfill(4))
        == {f"{value:04b}" for value in range(16)},
        "primary table covers all sixteen e/c/m/d masks",
        checks,
    )

    effects = pd.read_csv(
        ROOT / "results/factorial_v4_2x4/factorial_effects_bca.csv"
    )
    main_effects = effects[effects["effect_type"] == "main"]
    check(len(main_effects) == 48, "component table has 48 main effects", checks)
    check(
        set(main_effects["effect"])
        == {"mse", "correlation", "mmd", "derivative"},
        "component table contains all four components",
        checks,
    )

    smartwatch = pd.read_csv(
        ROOT / "results/factorial_v4/poster_evidence/selected_smartwatch.csv"
    )
    check(len(smartwatch) == 12, "smartwatch table has 12 family-device cells", checks)
    check(
        (smartwatch["r2"] < 0).all(),
        "all displayed smartwatch R2 values are negative",
        checks,
    )

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "checks_passed": sum(item["pass"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    (HERE / "VERIFICATION.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"{result['status']}: {result['checks_passed']}/{result['checks_total']}")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
