from __future__ import annotations

import json
from pathlib import Path

from scripts.launch_dependent_queue import upstream_ready


def write_state(path: Path, statuses: list[str]) -> None:
    path.write_text(json.dumps({"jobs": [{"status": status} for status in statuses]}))


def test_upstream_ready_requires_exact_completed_grid_and_output(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    marker = tmp_path / "complete.md"
    write_state(state, ["completed", "running"])
    marker.write_text("complete")

    ready, reason = upstream_ready(state, expected_jobs=2, required_output=marker)
    assert not ready
    assert "running=1" in reason

    write_state(state, ["completed", "completed"])
    ready, reason = upstream_ready(state, expected_jobs=2, required_output=marker)
    assert ready
    assert "2/2" in reason


def test_upstream_ready_rejects_wrong_job_count_or_empty_marker(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    marker = tmp_path / "complete.md"
    write_state(state, ["completed"])
    marker.write_text("")

    ready, reason = upstream_ready(state, expected_jobs=2, required_output=marker)
    assert not ready
    assert "1/2" in reason

    ready, reason = upstream_ready(state, expected_jobs=1, required_output=marker)
    assert not ready
    assert "missing" in reason
