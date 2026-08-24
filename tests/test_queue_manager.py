import importlib.util
import os
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / ".agents"
    / "skills"
    / "experiment-queue"
    / "scripts"
    / "queue_manager.py"
)
SPEC = importlib.util.spec_from_file_location("queue_manager", MODULE_PATH)
queue_manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue_manager)


def _job(expected_output, started):
    return {
        "id": "test",
        "screen_name": "EQ_test",
        "pid": None,
        "expected_output": expected_output,
        "started": started,
    }


def test_output_exists_supports_globs(tmp_path):
    (tmp_path / "artifact.json").write_text("{}")
    assert queue_manager.output_exists("*.json", str(tmp_path))


def test_live_screen_does_not_complete_from_existing_output(tmp_path, monkeypatch):
    (tmp_path / "artifact.json").write_text("{}")
    monkeypatch.setattr(queue_manager, "screen_exists", lambda _: True)
    monkeypatch.setattr(queue_manager, "detect_oom_in_log", lambda _: False)
    status, _ = queue_manager.job_status_check(
        _job("artifact.json", datetime.now(timezone.utc).isoformat()),
        str(tmp_path),
        str(tmp_path),
    )
    assert status == "running"


def test_success_requires_fresh_expected_output(tmp_path, monkeypatch):
    output = tmp_path / "artifact.json"
    output.write_text("{}")
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    os.utime(output, (old.timestamp(), old.timestamp()))
    (tmp_path / "test.log.exitcode").write_text("0")
    monkeypatch.setattr(queue_manager, "screen_exists", lambda _: False)
    monkeypatch.setattr(queue_manager, "detect_oom_in_log", lambda _: False)

    status, error = queue_manager.job_status_check(
        _job("artifact.json", datetime.now(timezone.utc).isoformat()),
        str(tmp_path),
        str(tmp_path),
    )
    assert status == "failed_other"
    assert "not refreshed" in error


def test_success_accepts_output_refreshed_by_attempt(tmp_path, monkeypatch):
    started = datetime.now(timezone.utc) - timedelta(seconds=2)
    (tmp_path / "artifact.json").write_text("{}")
    (tmp_path / "test.log.exitcode").write_text("0")
    monkeypatch.setattr(queue_manager, "screen_exists", lambda _: False)
    monkeypatch.setattr(queue_manager, "detect_oom_in_log", lambda _: False)

    status, error = queue_manager.job_status_check(
        _job("artifact.json", started.isoformat()),
        str(tmp_path),
        str(tmp_path),
    )
    assert status == "completed"
    assert error is None


def test_nonzero_exitcode_cannot_be_masked_by_output(tmp_path, monkeypatch):
    started = datetime.now(timezone.utc) - timedelta(seconds=2)
    (tmp_path / "artifact.json").write_text("{}")
    (tmp_path / "test.log.exitcode").write_text("7")
    monkeypatch.setattr(queue_manager, "screen_exists", lambda _: False)
    monkeypatch.setattr(queue_manager, "detect_oom_in_log", lambda _: False)

    status, error = queue_manager.job_status_check(
        _job("artifact.json", started.isoformat()),
        str(tmp_path),
        str(tmp_path),
    )
    assert status == "failed_other"
    assert "code 7" in error


def test_allowlisted_transient_cuda_fault_is_retryable(tmp_path, monkeypatch):
    (tmp_path / "test.log").write_text(
        "CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error\n"
    )
    (tmp_path / "test.log.exitcode").write_text("134")
    monkeypatch.setattr(queue_manager, "screen_exists", lambda _: False)
    monkeypatch.setattr(queue_manager, "detect_oom_in_log", lambda _: False)

    status, error = queue_manager.job_status_check(
        _job(None, datetime.now(timezone.utc).isoformat()),
        str(tmp_path),
        str(tmp_path),
    )

    assert status == "failed_transient"
    assert "hardware/driver" in error


def test_device_side_assert_remains_nonretryable(tmp_path, monkeypatch):
    (tmp_path / "test.log").write_text("CUDA error: device-side assert triggered\n")
    (tmp_path / "test.log.exitcode").write_text("134")
    monkeypatch.setattr(queue_manager, "screen_exists", lambda _: False)
    monkeypatch.setattr(queue_manager, "detect_oom_in_log", lambda _: False)

    status, error = queue_manager.job_status_check(
        _job(None, datetime.now(timezone.utc).isoformat()),
        str(tmp_path),
        str(tmp_path),
    )

    assert status == "failed_other"
    assert "code 134" in error


def test_stuck_job_does_not_complete_dependency_phase():
    state = {
        "jobs": [
            {"phase": "train", "status": "completed"},
            {"phase": "train", "status": "stuck"},
        ]
    }
    assert not queue_manager.phase_complete("train", state)


def test_assign_jobs_synchronizes_new_manifest_phases():
    state = {
        "phases": [
            {"name": "train", "depends_on": [], "status": "completed"},
        ],
        "jobs": [
            {
                "id": "train_one",
                "phase": "train",
                "cmd": "python train.py",
                "expected_output": "model.pt",
                "status": "completed",
            },
        ],
    }
    manifest = {
        "phases": [
            {
                "name": "train",
                "depends_on": [],
                "jobs": [
                    {
                        "id": "train_one",
                        "cmd": "python train.py",
                        "expected_output": "model.pt",
                    },
                ],
            },
            {
                "name": "evaluate",
                "depends_on": ["train"],
                "jobs": [
                    {
                        "id": "evaluate_one",
                        "cmd": "python evaluate.py",
                        "expected_output": "metrics.json",
                    },
                ],
            },
        ],
    }

    queue_manager.assign_jobs_to_phases(manifest, state)

    assert state["phases"] == [
        {"name": "train", "depends_on": [], "status": "completed"},
        {
            "name": "evaluate",
            "depends_on": ["train"],
            "status": "pending",
        },
    ]
    assert state["jobs"][-1]["id"] == "evaluate_one"
    assert state["jobs"][-1]["status"] == "pending"


def test_disk_guard_cleans_during_low_space(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    readings = iter(
        [
            usage(100 * 2**30, 98 * 2**30, 2 * 2**30),
            usage(100 * 2**30, 94 * 2**30, 6 * 2**30),
        ]
    )
    calls = []
    monkeypatch.setattr(queue_manager.shutil, "disk_usage", lambda _: next(readings))
    monkeypatch.setattr(
        queue_manager,
        "run",
        lambda cmd: (calls.append(cmd) or "", 0),
    )
    state = {"meta": {}}
    manifest = {
        "minimum_free_disk_gib": 4,
        "low_disk_cleanup_cmd": "git prune --expire=15.minutes.ago",
    }

    assert queue_manager.disk_space_guard(manifest, state, str(tmp_path))
    assert calls and "git prune" in calls[0]
    assert state["meta"]["disk_guard"]["status"] == "healthy"


def test_disk_guard_blocks_launches_if_cleanup_cannot_restore_space(
    tmp_path, monkeypatch
):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        queue_manager.shutil,
        "disk_usage",
        lambda _: usage(100 * 2**30, 98 * 2**30, 2 * 2**30),
    )
    monkeypatch.setattr(queue_manager, "run", lambda _: ("", 0))
    state = {"meta": {}}
    manifest = {
        "minimum_free_disk_gib": 4,
        "low_disk_cleanup_cmd": "git prune --expire=15.minutes.ago",
    }

    assert not queue_manager.disk_space_guard(manifest, state, str(tmp_path))
    assert (
        state["meta"]["disk_guard"]["status"]
        == "launches_paused_low_disk"
    )
