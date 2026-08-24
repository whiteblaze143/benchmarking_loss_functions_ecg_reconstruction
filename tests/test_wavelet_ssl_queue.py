from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import wavelet_ssl_queue as queue


FAKE_TRAINER = r'''
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--run-name",required=True);p.add_argument("--output-dir",required=True)
p.add_argument("--data-manifest",required=True);p.add_argument("--factorial-mask",required=True)
p.add_argument("--observed-leads",type=int,required=True);p.add_argument("--epochs",type=int,required=True)
p.add_argument("--seed",type=int,required=True);p.add_argument("--batch-size",type=int,required=True)
p.add_argument("--wavelet-bank",default="morlet");p.add_argument("--ssl-mode",default="none")
p.add_argument("--checkpoint-policy",default="none");p.add_argument("--quick-verify",action="store_true")
a=p.parse_args();root=Path.cwd();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
config={"run_name":a.run_name,"output_dir":a.output_dir,"data_manifest":a.data_manifest,
 "delineation_dir":None,"factorial_mask":a.factorial_mask,"observed_leads":[a.observed_leads],
 "epochs":1 if a.quick_verify else a.epochs,"seed":a.seed,"batch_size":a.batch_size,
 "wavelet_bank":a.wavelet_bank,"custom_wavelet_asset":None,"init_checkpoint":None,
 "ssl_mode":a.ssl_mode,"checkpoint_policy":a.checkpoint_policy,"quick_verify":a.quick_verify}
inputs={"trainer_sha256":h(__file__),
 "model_sha256":h(root/"unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py"),
 "data_manifest_sha256":h(a.data_manifest)}
training={k:v for k,v in config.items() if k!="quick_verify"}
raw=json.dumps({"config":training,"input_fingerprints":inputs},sort_keys=True,separators=(",",":"),allow_nan=False)
protocol=hashlib.sha256(raw.encode()).hexdigest();epochs=config["epochs"]
(out/"config.json").write_text(json.dumps(config,sort_keys=True)+"\n")
(out/"metrics.jsonl").write_text("".join(json.dumps({"epoch":i,"metric":float(i)})+"\n" for i in range(1,epochs+1)))
summary={"run_name":a.run_name,"epochs_completed":epochs,"training_config_sha256":protocol,
 "input_fingerprints":inputs,"metric":1.0}
(out/"summary.json").write_text(json.dumps(summary,sort_keys=True)+"\n")
success={"version":1,"run_name":a.run_name,"epochs_completed":epochs,
 "training_config_sha256":protocol,"training_config":training,"input_fingerprints":inputs,
 "checkpoint_artifacts":{},"config_sha256":h(out/"config.json"),
 "metrics_sha256":h(out/"metrics.jsonl"),"summary_sha256":h(out/"summary.json")}
(out/"_SUCCESS.json").write_text(json.dumps(success,sort_keys=True)+"\n")
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_manifest(tmp_path: Path, *, sleep: bool = False) -> tuple[Path, list[str]]:
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text("import time;time.sleep(60)\n" if sleep else FAKE_TRAINER)
    data_manifest = tmp_path / "data_manifest.json";data_manifest.write_text("{}\n")
    output = tmp_path / "output"
    command = [
        sys.executable, str(trainer), "--run-name", "job", "--output-dir", str(output),
        "--data-manifest", str(data_manifest), "--factorial-mask", "1000000",
        "--observed-leads", "0", "--epochs", "2", "--seed", "42",
        "--batch-size", "4", "--wavelet-bank", "morlet", "--ssl-mode", "both",
        "--checkpoint-policy", "none",
    ]
    manifest = {
        "version": 2, "cells": 1, "trainer_sha256": sha(trainer),
        "model_sha256": sha(Path("unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py")),
        "data_manifest_sha256": sha(data_manifest),
        "jobs": [{"id": "job", "command": command, "cell": {"ssl_mode": "both"}}],
    }
    path = tmp_path / "arbitrary_manifest_name.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return path, command


def patch_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"memory_used_mib": 0, "memory_free_mib": 40960, "utilization_pct": 0,
                "ecc_corrected": 10, "ecc_uncorrected": 0}
    monkeypatch.setattr(queue, "wait_for_resources", lambda *args, **kwargs: dict(snapshot))
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: dict(snapshot))
    real_sleep = time.sleep
    monkeypatch.setattr(queue.time, "sleep", lambda _: real_sleep(0.01))


def test_sqlite_queue_success_restart_and_arbitrary_manifest_name(tmp_path, monkeypatch):
    manifest, _ = setup_manifest(tmp_path);patch_resources(monkeypatch)
    assert queue.run_queue(manifest, project_root=Path.cwd(), job_timeout_seconds=10) == 0
    state = json.loads(manifest.with_suffix(".state.json").read_text())
    assert state["queue_status"] == "completed" and state["manifest_sha256"] == sha(manifest)
    with sqlite3.connect(tmp_path / "queue.sqlite") as connection:
        assert connection.execute("SELECT status,attempts FROM jobs").fetchone() == ("completed", 1)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert queue.run_queue(manifest, project_root=Path.cwd(), job_timeout_seconds=10) == 0
    with sqlite3.connect(tmp_path / "queue.sqlite") as connection:
        assert connection.execute("SELECT attempts FROM jobs").fetchone()[0] == 1


def test_changed_full_command_contract_rejects_stale_artifacts(tmp_path, monkeypatch):
    manifest, _ = setup_manifest(tmp_path);patch_resources(monkeypatch)
    assert queue.run_queue(manifest, project_root=Path.cwd(), job_timeout_seconds=10) == 0
    config_path = tmp_path / "output/config.json"
    config = json.loads(config_path.read_text());config["ssl_mode"] = "local"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    assert queue.run_queue(manifest, project_root=Path.cwd(), job_timeout_seconds=10) == 1
    with sqlite3.connect(tmp_path / "queue.sqlite") as connection:
        assert connection.execute("SELECT status FROM jobs").fetchone()[0] == "failed"


def test_stop_is_paused_not_success(tmp_path, monkeypatch):
    manifest, _ = setup_manifest(tmp_path);patch_resources(monkeypatch)
    (tmp_path / "STOP").write_text("pause\n")
    assert queue.run_queue(manifest, project_root=Path.cwd()) == 75
    state = json.loads(manifest.with_suffix(".state.json").read_text())
    assert state["queue_status"] == "paused"
    assert not (tmp_path / "_QUEUE_SUCCESS.json").exists()


def test_view_specific_custom_asset_is_fingerprinted_and_manifested(tmp_path):
    manifest_path, command = setup_manifest(tmp_path)
    asset = tmp_path / "ueg.pt"
    asset.write_bytes(b"physiology-derived-bank")
    command.extend([
        "--view-a-bank", "morlet", "--view-b-bank", "custom_asset",
        "--view-b-custom-wavelet-asset", str(asset),
    ])
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"][0]["command"] = command
    manifest["custom_wavelet_assets"] = {str(asset.resolve()): sha(asset)}
    config = {
        "data_manifest": str(tmp_path / "data_manifest.json"),
        "wavelet_bank": "morlet", "custom_wavelet_asset": None,
        "view_a_bank": "morlet", "view_b_bank": "custom_asset",
        "view_a_custom_wavelet_asset": None,
        "view_b_custom_wavelet_asset": str(asset), "init_checkpoint": None,
    }
    fingerprints = queue.current_input_fingerprints(config, command, Path.cwd())
    assert fingerprints["view_b_custom_wavelet_asset_sha256"] == sha(asset)
    queue.verify_manifest_inputs(manifest, manifest["jobs"], Path.cwd())
    asset.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="custom-wavelet inventory"):
        queue.verify_manifest_inputs(manifest, manifest["jobs"], Path.cwd())


def test_live_orphan_fails_closed_without_scheduling_duplicate(tmp_path, monkeypatch):
    manifest, command = setup_manifest(tmp_path, sleep=True);patch_resources(monkeypatch)
    connection = queue.connect(tmp_path / "queue.sqlite");queue.initialize(connection, manifest, Path.cwd())
    child = subprocess.Popen(command, cwd=Path.cwd(), start_new_session=True)
    try:
        real_sleep = time.sleep;real_sleep(0.05)
        connection.execute("UPDATE jobs SET status='running',attempts=1,child_pid=? WHERE id='job'", (child.pid,))
        connection.commit();connection.close()
        assert queue.run_queue(manifest, project_root=Path.cwd()) == 74
        with sqlite3.connect(tmp_path / "queue.sqlite") as check:
            assert check.execute("SELECT status,attempts FROM jobs").fetchone() == ("running", 1)
    finally:
        child.terminate();child.wait(timeout=5)
