from pathlib import Path

from scripts.verify_production_ready import _tracker_status, verify


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_tracker_parser_reads_current_gate_states() -> None:
    statuses = _tracker_status(EXPERIMENT / "refine-logs/EXPERIMENT_TRACKER.md")
    assert statuses["R001"] == "TODO"
    assert statuses["R023"] == "IN_PROGRESS"


def test_current_incomplete_program_cannot_create_readiness_sentinel() -> None:
    result = verify(EXPERIMENT)
    assert result["status"] == "BLOCKED"
    assert any("gate R023" in error for error in result["errors"])
    assert any("paper input reconciliation is blocked" in error for error in result["errors"])
    assert any("missing production results database" in error for error in result["errors"])
    assert not (EXPERIMENT / "outputs/PRODUCTION_READY").exists()
