import json

import pytest
import torch

from scripts.infer_factorial_checkpoint import atomic_torch_save, compatible_identities


def _write_contract(path, *, contract_id="contract-v1", source="source-v1"):
    path.write_text(
        json.dumps(
            {
                "contract_id": contract_id,
                "approved_source_bundle_sha256": source,
            }
        )
    )


def _write_audit(path, *, contract_id="contract-v1", source="source-v1"):
    path.write_text(
        json.dumps(
            {
                "contract": {
                    "contract_id": contract_id,
                    "approved_source_bundle_sha256": source,
                },
                "models": [
                    {
                        "model_id": "f_1000000_s42",
                        "compatible": True,
                        "checkpoint_sha256": "exact-digest",
                        "source_bundle_sha256": source,
                    },
                    {
                        "model_id": "f_legacy_s42",
                        "compatible": False,
                        "checkpoint_sha256": "legacy-digest",
                        "source_bundle_sha256": source,
                    },
                ],
            }
        )
    )


def test_compatible_identities_returns_only_current_approved_rows(tmp_path):
    contract = tmp_path / "contract.json"
    audit = tmp_path / "audit.json"
    _write_contract(contract)
    _write_audit(audit)
    ready = compatible_identities(audit, contract)
    assert list(ready) == ["f_1000000_s42"]
    assert ready["f_1000000_s42"]["checkpoint_sha256"] == "exact-digest"


def test_compatible_identities_rejects_stale_contract(tmp_path):
    contract = tmp_path / "contract.json"
    audit = tmp_path / "audit.json"
    _write_contract(contract)
    _write_audit(audit, contract_id="old-contract")
    with pytest.raises(RuntimeError, match="current contract"):
        compatible_identities(audit, contract)


def test_compatible_identities_rejects_wrong_approved_source(tmp_path):
    contract = tmp_path / "contract.json"
    audit = tmp_path / "audit.json"
    _write_contract(contract)
    _write_audit(audit, source="wrong-source")
    with pytest.raises(RuntimeError, match="approved source bundle"):
        compatible_identities(audit, contract)


def test_atomic_torch_save_round_trips_without_temporary_file(tmp_path):
    destination = tmp_path / "nested" / "reconstruction.pt"
    expected = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    atomic_torch_save(expected, destination)
    observed = torch.load(destination, map_location="cpu", weights_only=True)
    assert torch.equal(observed, expected)
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_atomic_torch_save_removes_temporary_after_failure(tmp_path, monkeypatch):
    destination = tmp_path / "reconstruction.pt"

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="injected save failure"):
        atomic_torch_save(torch.ones(1), destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
