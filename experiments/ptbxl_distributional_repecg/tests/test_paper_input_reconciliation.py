import json
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_current_shared_input_misuse_is_explicitly_blocked() -> None:
    payload = json.loads((EXPERIMENT / "outputs/paper_input_reconciliation.json").read_text())
    assert payload["status"] == "blocked"
    complete = {1, 2, 3, 4}
    paper05_manifest = Path(
        "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
        "paper05_koopman/development_representations/manifest.json"
    )
    if paper05_manifest.is_file():
        complete.add(5)
    paper06_manifest = Path(
        "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
        "paper06_conditional/development_representations/manifest.json"
    )
    if paper06_manifest.is_file() and json.loads(paper06_manifest.read_text()).get("status") == "complete":
        complete.add(6)
    assert payload["blocked_papers"] == [
        f"paper{paper_id:02d}" for paper_id in range(1, 16) if paper_id not in complete
    ]
    assert payload["papers"]["paper01"]["status"] == "complete"
    assert payload["papers"]["paper02"]["status"] == "complete"
    assert payload["papers"]["paper03"]["status"] == "complete"
    assert payload["papers"]["paper04"]["status"] == "ineligible_nystrom_fidelity"
    assert payload["papers"]["paper05"]["required_input"] == (
        "record_specific_regularized_koopman_descriptors"
    )


def test_smoke_checks_input_reconciliation_before_database_reset() -> None:
    source = (EXPERIMENT / "scripts/run_smoke_master.sh").read_text()
    gate = source.index("input_reconciliation=")
    reset = source.index('rm -f -- "$results_db"')
    assert gate < reset
