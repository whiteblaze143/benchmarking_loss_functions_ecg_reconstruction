import json
import contextlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
import torch
import torch.nn as nn

import scripts.checkpoint_store as checkpoint_store
from scripts.checkpoint_store import (
    catalog_completed,
    connect,
    enrich_sidecar,
    load_checkpoint,
    materialize,
    prune_cache,
    set_settings,
    status,
)
from scripts.mixed_factorial_release import (
    EXPECTED_GATE_IDS,
    build_release_table,
    cohort_order_sha256,
    expected_models,
    sha256_file,
    validate_checkpoint_catalog,
)
from scripts.train_factorial import split_content_root
from scripts.train_mcma_3lead import MCMAModel


def test_archive_watcher_releases_writer_lock_before_sleep(tmp_path, monkeypatch):
    events = []

    @contextlib.contextmanager
    def recording_lock(_db_path):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def stop_after_first_pass(_seconds):
        assert events[-1] == "exit"
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint_store, "store_lock", recording_lock)
    monkeypatch.setattr(
        checkpoint_store,
        "archive_completed_pass",
        lambda _connection, _args: events.append("archive"),
    )
    monkeypatch.setattr(checkpoint_store.time, "sleep", stop_after_first_pass)
    args = SimpleNamespace(
        db=tmp_path / "catalog.sqlite",
        repo="owner/repository",
        tag="checkpoints",
        poll_seconds=60,
    )
    with pytest.raises(KeyboardInterrupt):
        checkpoint_store.archive_watch(args)
    assert events == ["enter", "archive", "exit"]


def _state(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "f_1000000_s42",
                        "status": "completed",
                        "completed": "now",
                    }
                ]
            }
        )
    )
    return path


def test_catalog_maps_multiarchitecture_filename_to_queue_identity(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "factorial_msvae_1000000_s42.pt").write_bytes(b"checkpoint")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"jobs": [{
        "id": "msvae_f_1000000_s42",
        "status": "completed",
        "completed": "now",
    }]}))
    connection = connect(tmp_path / "catalog.sqlite")
    assert catalog_completed(connection, state, checkpoint_dir) == 1
    row = connection.execute("SELECT * FROM checkpoints").fetchone()
    assert row["model_id"] == "factorial_msvae_1000000_s42"
    assert row["factorial_mask"] == "1000000"
    assert row["seed"] == 42


def test_changed_generation_clears_remote_and_semantic_state(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "factorial_1000000_s42.pt"
    checkpoint.write_bytes(b"old generation")
    state = _state(tmp_path / "state.json")
    connection = connect(tmp_path / "catalog.sqlite")
    catalog_completed(connection, state, checkpoint_dir)
    old_digest = connection.execute(
        "SELECT sha256 FROM checkpoints"
    ).fetchone()["sha256"]
    connection.execute(
        """
        UPDATE checkpoints SET
            status='error', asset_id=9, asset_name='QUARANTINED_old.pt',
            asset_size_bytes=size_bytes, asset_digest='sha256:' || sha256,
            asset_state='uploaded', remote_verified_at='old',
            round_trip_verified_at='old', payload_validated_at='old',
            payload_tensor_count=162, payload_factorial_mask='1000000',
            payload_seed=42, payload_state_schema_sha256='stale'
        """
    )
    connection.commit()
    checkpoint.write_bytes(b"new retrained generation")
    catalog_completed(connection, state, checkpoint_dir)
    row = connection.execute("SELECT * FROM checkpoints").fetchone()
    assert row["sha256"] != old_digest
    assert row["status"] == "local"
    for field in (
        "asset_id",
        "asset_name",
        "asset_digest",
        "remote_verified_at",
        "round_trip_verified_at",
        "payload_validated_at",
        "payload_tensor_count",
        "payload_factorial_mask",
        "payload_seed",
        "payload_state_schema_sha256",
    ):
        assert row[field] is None


def test_tensor_content_root_detects_same_size_mutation(tmp_path):
    tensor = tmp_path / "1.pt"
    tensor.write_bytes(b"AAAA")
    before = split_content_root(tmp_path)
    tensor.write_bytes(b"ZZZZ")
    after = split_content_root(tmp_path)
    assert before["records"] == after["records"] == 1
    assert before["content_root_sha256"] != after["content_root_sha256"]


def test_canonical_test_cohort_joint_order_hash():
    root = Path(__file__).resolve().parents[1]
    metadata = pd.read_csv(
        root / "data/ptb_xl/ptbxl_database.csv",
        usecols=["ecg_id", "patient_id", "strat_fold"],
    )
    test = metadata.loc[metadata["strat_fold"].eq(10)].sort_values("ecg_id")
    assert len(test) == 2_198
    assert test["patient_id"].nunique() == 1_904
    assert cohort_order_sha256(test["ecg_id"], test["patient_id"]) == (
        "5f85303e675ae817e486e693dbaba9ca1dfa4892c473f53ca1b785b9937e241d"
    )


def test_error_asset_cannot_materialize(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "factorial_1000000_s42.pt"
    checkpoint.write_bytes(b"partial")
    state = _state(tmp_path / "state.json")
    connection = connect(tmp_path / "catalog.sqlite")
    catalog_completed(connection, state, checkpoint_dir)
    connection.execute(
        """
        UPDATE checkpoints SET
            status='error', asset_id=9, asset_name='QUARANTINED_partial.pt',
            local_path=NULL
        """
    )
    connection.commit()
    with pytest.raises(RuntimeError, match="not eligible"):
        materialize(connection, "f_1000000_s42", tmp_path / "cache")


def test_status_separates_inference_addressable_and_error_bytes(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "factorial_1000000_s42.pt"
    checkpoint.write_bytes(b"current-generation")
    connection = connect(tmp_path / "catalog.sqlite")
    catalog_completed(connection, _state(tmp_path / "state.json"), checkpoint_dir)
    current_size = checkpoint.stat().st_size
    connection.execute(
        """
        INSERT INTO checkpoints(
            model_id, factorial_mask, seed, filename, size_bytes, sha256,
            status, error, created_at, updated_at
        ) VALUES (
            'f_legacy_s42', 'legacy0', 42, 'legacy.pt', 123, 'legacy-digest',
            'error', 'quarantined historical generation', 'now', 'now'
        )
        """
    )
    connection.execute(
        "UPDATE checkpoints SET status='remote_verified' WHERE model_id='f_1000000_s42'"
    )
    connection.commit()

    observed = status(connection)

    assert observed["logical_bytes"] == current_size + 123
    assert observed["inference_addressable_bytes"] == current_size
    assert observed["error_generation_bytes"] == 123
    assert observed["local_bytes"] == current_size


def test_verified_remote_materialize_strict_forward_and_prune(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    source = checkpoint_dir / "factorial_1000000_s42.pt"
    model = MCMAModel(in_channels=3, out_channels=12)
    torch.save(
        {
            "model_state_dict": {
                key: value.to(torch.float16)
                for key, value in model.state_dict().items()
            },
            "provenance": {"factorial_mask": "1000000", "seed": 42},
        },
        source,
    )
    remote_bytes = tmp_path / "remote.pt"
    shutil.copy2(source, remote_bytes)
    db = tmp_path / "catalog.sqlite"
    connection = connect(db)
    set_settings(connection, "owner/repository", "checkpoints")
    catalog_completed(connection, _state(tmp_path / "state.json"), checkpoint_dir)
    row = connection.execute("SELECT * FROM checkpoints").fetchone()
    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "refine-logs/factorial_training_contract.json"
        ).read_text()
    )
    connection.execute(
        """
        UPDATE checkpoints SET
            status='remote_verified', local_path=NULL, asset_id=7,
            asset_size_bytes=size_bytes, asset_digest='sha256:' || sha256,
            asset_state='uploaded', remote_verified_at='now',
            round_trip_verified_at='now', payload_validated_at='now',
            payload_tensor_count=162, payload_factorial_mask=factorial_mask,
            payload_seed=seed, payload_state_schema_sha256=?
        """,
        (contract["state_schema_sha256"],),
    )
    connection.commit()
    connection.close()
    source.unlink()

    monkeypatch.setattr(
        checkpoint_store,
        "download_asset_to_file",
        lambda _repo, _asset_id, destination: shutil.copy2(
            remote_bytes, destination
        ),
    )
    cache = tmp_path / "cache"
    payload = load_checkpoint(
        "f_1000000_s42", db_path=db, cache_dir=cache, map_location="cpu"
    )
    restored = MCMAModel(in_channels=3, out_channels=12)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        output = restored(torch.zeros(1, 3, 64))
    assert output.shape == (1, 12, 64)
    assert torch.isfinite(output).all()

    connection = connect(db)
    assert prune_cache(connection, cache, 0) == 1
    final = connection.execute("SELECT * FROM checkpoints").fetchone()
    connection.close()
    assert final["status"] == "remote_verified"
    assert final["local_path"] is None
    assert not list(cache.glob("factorial_*.pt"))


def test_materialize_rejects_generation_swap_during_download(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    source = checkpoint_dir / "factorial_1000000_s42.pt"
    source.write_bytes(b"old exact generation")
    remote_bytes = tmp_path / "remote.pt"
    shutil.copy2(source, remote_bytes)
    state = _state(tmp_path / "state.json")
    db = tmp_path / "catalog.sqlite"
    connection = connect(db)
    set_settings(connection, "owner/repository", "checkpoints")
    catalog_completed(connection, state, checkpoint_dir)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "refine-logs/factorial_training_contract.json"
        ).read_text()
    )
    connection.execute(
        """
        UPDATE checkpoints SET
            status='remote_verified', local_path=NULL, asset_id=7,
            asset_size_bytes=size_bytes, asset_digest='sha256:' || sha256,
            asset_state='uploaded', remote_verified_at='now',
            round_trip_verified_at='now', payload_validated_at='now',
            payload_tensor_count=162, payload_factorial_mask=factorial_mask,
            payload_seed=seed, payload_state_schema_sha256=?
        """,
        (contract["state_schema_sha256"],),
    )
    connection.commit()
    old_digest = connection.execute(
        "SELECT sha256 FROM checkpoints"
    ).fetchone()["sha256"]
    source.unlink()

    def swap_generation(_repo, _asset_id, destination):
        shutil.copy2(remote_bytes, destination)
        source.write_bytes(b"new exact generation with another digest")
        other = connect(db)
        try:
            new_digest = checkpoint_store.sha256_file(source)
            other.execute(
                """
                UPDATE checkpoints SET
                    size_bytes=?, sha256=?, status='local', local_path=?,
                    asset_id=NULL, asset_name=NULL, asset_size_bytes=NULL,
                    asset_digest=NULL, asset_state=NULL,
                    remote_verified_at=NULL, round_trip_verified_at=NULL,
                    payload_validated_at=NULL
                WHERE model_id='f_1000000_s42'
                """,
                (source.stat().st_size, new_digest, str(source)),
            )
            other.commit()
        finally:
            other.close()

    monkeypatch.setattr(
        checkpoint_store, "download_asset_to_file", swap_generation
    )
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="generation changed"):
        materialize(connection, "f_1000000_s42", cache)
    current = connection.execute("SELECT * FROM checkpoints").fetchone()
    connection.close()
    assert current["sha256"] != old_digest
    assert current["status"] == "local"
    assert not list(cache.glob("factorial_*.pt"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_key", "Missing key"),
        ("wrong_shape", "size mismatch"),
        ("non_finite", "non-finite"),
        ("unsupported_dtype", "unsupported state dtypes"),
    ],
)
def test_semantic_validator_rejects_adversarial_states(
    tmp_path, monkeypatch, mutation, message
):
    class TinyModel(nn.Module):
        def __init__(self, in_channels=3, out_channels=12):
            super().__init__()
            self.projection = nn.Linear(in_channels, out_channels)

    monkeypatch.setattr(
        "scripts.train_mcma_3lead.MCMAModel",
        TinyModel,
    )
    monkeypatch.setattr(
        checkpoint_store,
        "DEFAULT_CHECKPOINT_DIR",
        tmp_path / "sidecars",
    )
    (tmp_path / "sidecars").mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "factorial_1000000_s42.pt"
    checkpoint.write_bytes(b"catalog placeholder")
    connection = connect(tmp_path / "catalog.sqlite")
    catalog_completed(
        connection,
        _state(tmp_path / "state.json"),
        checkpoint_dir,
    )
    row = connection.execute("SELECT * FROM checkpoints").fetchone()
    state = TinyModel().state_dict()
    if mutation == "missing_key":
        state.pop("projection.bias")
    elif mutation == "wrong_shape":
        state["projection.weight"] = torch.zeros(1, 1)
    elif mutation == "non_finite":
        state["projection.weight"][0, 0] = torch.nan
    elif mutation == "unsupported_dtype":
        state["projection.weight"] = state["projection.weight"].to(torch.int64)
    torch.save(
        {
            "model_state_dict": state,
            "provenance": {"factorial_mask": "1000000", "seed": 42},
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match=message):
        enrich_sidecar(connection, row, checkpoint)


def test_release_writer_rejects_all_true_undersized_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    rows = pd.DataFrame(
        sorted(expected_models(manifest)),
        columns=["model_id", "factorial_mask", "seed"],
    )
    summary = tmp_path / "summary.csv"
    rows.head(1).to_csv(summary, index=False)
    undersized = rows.head(1).copy()
    undersized["checkpoint_sha256"] = "0" * 64
    undersized["checkpoint_size_bytes"] = 1
    undersized["missing_mse"] = 0.1
    undersized["missing_pearson"] = 0.5
    undersized.to_csv(summary, index=False)
    per_record_manifest = tmp_path / "per_record_manifest.json"
    per_record_manifest.write_text("{}")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": sha256_file(manifest),
                "summary_sha256": sha256_file(summary),
                "compatibility_audit_sha256": sha256_file(
                    root / "results/checkpoint_store/compatibility_audit.json"
                ),
                "training_contract_sha256": sha256_file(
                    root / "refine-logs/factorial_training_contract.json"
                ),
                "checkpoint_catalog_sha256": sha256_file(
                    root / "results/checkpoint_store/catalog.sqlite"
                ),
                "per_record_manifest_sha256": sha256_file(per_record_manifest),
                "gates": [
                    {"gate_id": gate_id, "gate": gate_id, "pass": True}
                    for gate_id in sorted(EXPECTED_GATE_IDS)
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="all 480"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            per_record_manifest_path=per_record_manifest,
        )


def _all_true_report(
    path, manifest, summary, compatibility, per_record, checkpoint_catalog=None
):
    if checkpoint_catalog is None:
        checkpoint_catalog = (
            Path(__file__).resolve().parents[1]
            / "results/checkpoint_store/catalog.sqlite"
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": sha256_file(manifest),
                "summary_sha256": sha256_file(summary),
                "compatibility_audit_sha256": sha256_file(compatibility),
                "training_contract_sha256": sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "refine-logs/factorial_training_contract.json"
                ),
                "checkpoint_catalog_sha256": sha256_file(checkpoint_catalog),
                "per_record_manifest_sha256": sha256_file(per_record),
                "gates": [
                    {"gate_id": gate_id, "gate": gate_id, "pass": True}
                    for gate_id in sorted(EXPECTED_GATE_IDS)
                ],
            }
        )
    )


def test_release_writer_rejects_bound_incompatible_audit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    identities = sorted(expected_models(manifest))
    summary = tmp_path / "summary.csv"
    pd.DataFrame(
        identities, columns=["model_id", "factorial_mask", "seed"]
    ).assign(
        checkpoint_sha256="0" * 64,
        checkpoint_size_bytes=1,
        missing_mse=0.1,
        missing_pearson=0.5,
    ).to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counts": {"incompatible": 480},
                "models": [
                    {
                        "model_id": model_id,
                        "factorial_mask": mask,
                        "seed": seed,
                        "compatible": False,
                    }
                    for model_id, mask, seed in identities
                ],
            }
        )
    )
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(json.dumps({"models": []}))
    report = tmp_path / "report.json"
    _all_true_report(
        report, manifest, summary, compatibility, per_record
    )
    with pytest.raises(ValueError, match="incompatible"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            compatibility_audit_path=compatibility,
            per_record_manifest_path=per_record,
        )


def _valid_compatibility_payload(root, identities):
    contract = json.loads(
        (root / "refine-logs/factorial_training_contract.json").read_text()
    )
    return {
        "schema_version": 1,
        "counts": {"compatible": 480},
        "contract": contract,
        "models": [
            {
                "model_id": model_id,
                "factorial_mask": mask,
                "seed": seed,
                "compatible": True,
                "checkpoint_sha256": "0" * 64,
                "checkpoint_size_bytes": 1,
                "source_bundle_sha256": contract[
                    "approved_source_bundle_sha256"
                ],
                "source_policy": "pinned-current-exact-sources",
                "embedded_sources_valid": True,
                "state_precision": ["torch.float16"],
                "state_schema_sha256": contract["state_schema_sha256"],
                "data_content_compatible": True,
                "reasons": [],
            }
            for model_id, mask, seed in identities
        ],
    }


def _full_summary(identities):
    return pd.DataFrame(
        identities, columns=["model_id", "factorial_mask", "seed"]
    ).assign(
        checkpoint_sha256="0" * 64,
        checkpoint_size_bytes=1,
        missing_mse=0.1,
        missing_pearson=0.5,
    )


def test_release_writer_rejects_bound_empty_per_record_manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    identities = sorted(expected_models(manifest))
    summary = tmp_path / "summary.csv"
    _full_summary(identities).to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(_valid_compatibility_payload(root, identities))
    )
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(json.dumps({"models": []}))
    report = tmp_path / "report.json"
    _all_true_report(report, manifest, summary, compatibility, per_record)
    with pytest.raises(ValueError, match="exact 480 model ids"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            compatibility_audit_path=compatibility,
            per_record_manifest_path=per_record,
        )


def test_release_writer_rejects_reused_per_record_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    identities = sorted(expected_models(manifest))
    summary = tmp_path / "summary.csv"
    _full_summary(identities).to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(_valid_compatibility_payload(root, identities))
    )
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": model_id,
                        "factorial_mask": mask,
                        "seed": seed,
                        "status": "complete",
                        "path": "results/factorial_mixed_level/per_record/shared.parquet",
                        "checkpoint_sha256": "0" * 64,
                        "checkpoint_size_bytes": 1,
                    }
                    for model_id, mask, seed in identities
                ]
            }
        )
    )
    report = tmp_path / "report.json"
    _all_true_report(report, manifest, summary, compatibility, per_record)
    with pytest.raises(ValueError, match="480 distinct artifact paths"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            compatibility_audit_path=compatibility,
            per_record_manifest_path=per_record,
        )


def test_release_writer_rejects_canonicalized_path_aliases(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    identities = sorted(expected_models(manifest))
    summary = tmp_path / "summary.csv"
    _full_summary(identities).to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(_valid_compatibility_payload(root, identities))
    )
    aliases = (
        "results/factorial_mixed_level/per_record/shared.parquet",
        "results/factorial_mixed_level/per_record/sub/../shared.parquet",
    )
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": model_id,
                        "factorial_mask": mask,
                        "seed": seed,
                        "status": "complete",
                        "path": aliases[index % 2],
                        "checkpoint_sha256": "0" * 64,
                        "checkpoint_size_bytes": 1,
                    }
                    for index, (model_id, mask, seed) in enumerate(identities)
                ]
            }
        )
    )
    report = tmp_path / "report.json"
    _all_true_report(report, manifest, summary, compatibility, per_record)
    with pytest.raises(ValueError, match="480 distinct artifact paths"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            compatibility_audit_path=compatibility,
            per_record_manifest_path=per_record,
        )


def test_catalog_validator_rejects_semantic_identity_mismatch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    contract_path = root / "refine-logs/factorial_training_contract.json"
    contract = json.loads(contract_path.read_text())
    identities = sorted(expected_models(manifest))
    db = tmp_path / "catalog.sqlite"
    connection = connect(db)
    for model_id, mask, seed in identities:
        metadata = {
            "schema_version": 3,
            "run_name": model_id,
            "factorial_mask": mask,
            "seed": seed,
            "checkpoint_size_bytes": 1,
            "checkpoint_sha256": "0" * 64,
            "state_tensor_count": 162,
            "state_schema_sha256": contract["state_schema_sha256"],
            "training_contract": contract,
        }
        connection.execute(
            """
            INSERT INTO checkpoints(
                model_id, factorial_mask, seed, filename, size_bytes, sha256,
                metadata_json, release_tag, asset_id, asset_name,
                asset_size_bytes, asset_digest, asset_state,
                round_trip_verified_at, payload_validated_at,
                payload_tensor_count, payload_factorial_mask, payload_seed,
                payload_state_schema_sha256, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, 'tag', 1, ?, 1, ?, 'uploaded',
                      'now', 'now', 162, ?, ?, ?, 'remote_verified', 'now', 'now')
            """,
            (
                model_id,
                mask,
                seed,
                f"factorial_{mask}_s{seed}.pt",
                "0" * 64,
                json.dumps(metadata),
                f"factorial_{mask}_s{seed}.pt",
                "sha256:" + "0" * 64,
                mask,
                seed,
                contract["state_schema_sha256"],
            ),
        )
    connection.execute(
        "UPDATE checkpoints SET payload_seed=999 WHERE model_id=?",
        (identities[0][0],),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="catalog generation mismatch"):
        validate_checkpoint_catalog(
            db,
            set(identities),
            {model_id: ("0" * 64, 1) for model_id, _, _ in identities},
            contract_path,
        )


def test_release_writer_accepts_fully_bound_480_model_fixture(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    contract_path = root / "refine-logs/factorial_training_contract.json"
    contract = json.loads(contract_path.read_text())
    identities = sorted(expected_models(manifest))

    summary = tmp_path / "summary.csv"
    _full_summary(identities).to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(_valid_compatibility_payload(root, identities))
    )

    cohort = pd.read_csv(
        root / "data/ptb_xl/ptbxl_database.csv",
        usecols=["ecg_id", "patient_id", "strat_fold"],
    )
    cohort = cohort.loc[cohort["strat_fold"].eq(10)].sort_values("ecg_id")
    record_ids = cohort["ecg_id"].to_numpy()
    patient_ids = cohort["patient_id"].to_numpy()
    per_record_root = tmp_path / "per_record"
    per_record_root.mkdir()
    model_by_path = {}
    entries = []
    for model_id, mask, seed in identities:
        artifact = per_record_root / f"{model_id}.parquet"
        artifact.write_bytes(b"fixture")
        model_by_path[artifact.resolve()] = model_id
        entries.append(
            {
                "model_id": model_id,
                "factorial_mask": mask,
                "seed": seed,
                "status": "complete",
                "path": str(artifact.resolve()),
                "file_sha256": sha256_file(artifact),
                "record_order_sha256": (
                    "5f85303e675ae817e486e693dbaba9ca1dfa4892c473f53ca1b785b9937e241d"
                ),
                "checkpoint_sha256": "0" * 64,
                "checkpoint_size_bytes": 1,
            }
        )
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(json.dumps({"models": entries}))

    def fixture_parquet(path, columns=None):
        model_id = model_by_path[Path(path).resolve()]
        frame = pd.DataFrame(
            {
                "model_id": model_id,
                "checkpoint_sha256": "0" * 64,
                "record_id": record_ids,
                "patient_id": patient_ids,
                "missing_mse": 0.1,
                "missing_pearson": 0.5,
            }
        )
        return frame[list(columns)] if columns is not None else frame

    monkeypatch.setattr(
        "scripts.mixed_factorial_release.pd.read_parquet", fixture_parquet
    )

    db = tmp_path / "catalog.sqlite"
    connection = connect(db)
    for model_id, mask, seed in identities:
        metadata = {
            "schema_version": 3,
            "run_name": model_id,
            "factorial_mask": mask,
            "seed": seed,
            "checkpoint_size_bytes": 1,
            "checkpoint_sha256": "0" * 64,
            "state_tensor_count": 162,
            "state_schema_sha256": contract["state_schema_sha256"],
            "training_contract": contract,
        }
        connection.execute(
            """
            INSERT INTO checkpoints(
                model_id, factorial_mask, seed, filename, size_bytes, sha256,
                metadata_json, release_tag, asset_id, asset_name,
                asset_size_bytes, asset_digest, asset_state,
                round_trip_verified_at, payload_validated_at,
                payload_tensor_count, payload_factorial_mask, payload_seed,
                payload_state_schema_sha256, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, 'tag', 1, ?, 1, ?, 'uploaded',
                      'now', 'now', 162, ?, ?, ?, 'remote_verified', 'now', 'now')
            """,
            (
                model_id,
                mask,
                seed,
                f"factorial_{mask}_s{seed}.pt",
                "0" * 64,
                json.dumps(metadata),
                f"factorial_{mask}_s{seed}.pt",
                "sha256:" + "0" * 64,
                mask,
                seed,
                contract["state_schema_sha256"],
            ),
        )
    connection.commit()
    connection.close()

    report = tmp_path / "report.json"
    _all_true_report(
        report, manifest, summary, compatibility, per_record, db
    )
    output = tmp_path / "release.csv"
    assert build_release_table(
        summary,
        report,
        output,
        manifest_path=manifest,
        compatibility_audit_path=compatibility,
        per_record_manifest_path=per_record,
        training_contract_path=contract_path,
        checkpoint_catalog_path=db,
    ) == output
    released = pd.read_csv(output, dtype={"factorial_mask": str})
    assert len(released) == 480
    assert set(released["model_id"]) == {row[0] for row in identities}


def test_release_writer_rejects_nan_summary_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    identities = sorted(expected_models(manifest))
    frame = _full_summary(identities)
    frame.loc[0, "missing_mse"] = np.nan
    summary = tmp_path / "summary.csv"
    frame.to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(json.dumps({"schema_version": 1, "models": []}))
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(json.dumps({"models": []}))
    report = tmp_path / "report.json"
    _all_true_report(report, manifest, summary, compatibility, per_record)
    with pytest.raises(ValueError, match="metrics must all be finite"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            compatibility_audit_path=compatibility,
            per_record_manifest_path=per_record,
        )


def test_release_writer_rejects_unregistered_extra_result_column(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "refine-logs/factorial_manifest.json"
    identities = sorted(expected_models(manifest))
    frame = _full_summary(identities)
    frame["unvalidated_headline_metric"] = np.nan
    summary = tmp_path / "summary.csv"
    frame.to_csv(summary, index=False)
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(json.dumps({"schema_version": 1, "models": []}))
    per_record = tmp_path / "per_record_manifest.json"
    per_record.write_text(json.dumps({"models": []}))
    report = tmp_path / "report.json"
    _all_true_report(report, manifest, summary, compatibility, per_record)
    with pytest.raises(ValueError, match="exact registered release schema"):
        build_release_table(
            summary,
            report,
            tmp_path / "must_not_exist.csv",
            manifest_path=manifest,
            compatibility_audit_path=compatibility,
            per_record_manifest_path=per_record,
        )
