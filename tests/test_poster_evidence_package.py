from pathlib import Path

import pandas as pd

from scripts.build_poster_evidence_package import (
    build_claims,
    verify,
)


def test_claim_builder_keeps_publication_gate_explicit():
    architecture = pd.DataFrame([
        {
            "family": family,
            "ptbxl_r2": value,
            "echonext_r2": value,
        }
        for family, value in [("unet", 0.1), ("msvae", 0.2), ("ecgaim", 0.3)]
    ])
    effects = pd.DataFrame([
        {
            "family": family,
            "metric": "r2",
            "effect": "correlation",
            "estimate": value,
            "ci_low": value - 0.01,
            "ci_high": value + 0.01,
        }
        for family, value in [("unet", -0.1), ("msvae", 0.1), ("ecgaim", 0.1)]
    ])
    endpoints = pd.DataFrame([
        {
            "family": family,
            "endpoint": endpoint,
            "estimate": 0.1,
            "p_value": 0.01,
            "reject_null": endpoint != "diagnostic_utility",
        }
        for family in ("unet", "msvae", "ecgaim")
        for endpoint in ("QRS", "ST", "diagnostic_utility")
    ])
    robustness = pd.DataFrame([
        {
            "family": family,
            "condition": "nstdb_bw_0db",
            "missing_lead_mse": value,
        }
        for family, value in [("unet", 0.1), ("msvae", 0.2), ("ecgaim", 0.3)]
    ])
    smartwatch = pd.DataFrame({"r2": [-1.0, -0.5]})
    claims = build_claims(
        architecture, effects, endpoints, robustness, smartwatch
    )
    assert claims
    assert all(
        claim["status"] == "empirically_supported_pending_independent_audit"
        for claim in claims
    )
    assert "clinical validation" in claims[-1]["qualifiers"][0]


def test_verifier_rejects_changed_output(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("source")
    output = tmp_path / "package"
    output.mkdir()
    (output / "artifact.csv").write_text("value\n1\n")
    (output / "poster_claims.json").write_text(
        '[{"status":"empirically_supported_pending_independent_audit"}]'
    )
    from scripts.build_poster_evidence_package import sha256, strict_json

    strict_json(output / "evidence_manifest.json", {
        "source_sha256": {str(source): sha256(source)},
        "output_sha256": {"artifact.csv": sha256(output / "artifact.csv")},
    })
    # Absolute source paths are joined unchanged by pathlib.
    assert verify(output)["status"] == "PASS"
    (output / "artifact.csv").write_text("value\n2\n")
    try:
        verify(output)
    except RuntimeError as error:
        assert "output:artifact.csv" in str(error)
    else:
        raise AssertionError("Verifier accepted a modified artifact")
