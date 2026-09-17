import json
from pathlib import Path


MANIFEST = Path(
    "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
    "paper04_hankel_dynamics/development_representations/manifest.json"
)


def test_failed_nystrom_gate_is_preserved_and_fail_closed() -> None:
    payload = json.loads(MANIFEST.read_text())
    assert payload["status"] == "ineligible_nystrom_fidelity"
    assert set(payload["audits"]) == {"128", "256"}
    assert not payload["audits"]["128"]["passed"]
    assert not payload["audits"]["256"]["passed"]
    assert payload["audits"]["256"]["spearman"] < 0.9
    assert payload["audits"]["256"]["median_relative_error"] >= 0.15
