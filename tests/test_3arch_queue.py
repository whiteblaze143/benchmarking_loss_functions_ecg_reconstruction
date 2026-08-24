import json

import torch

import scripts.run_3arch_queue as queue
from scripts.reconcile_3arch_queue import restore_missing_architecture_pairs
from scripts.train_factorial_multimodel import atomic_torch_save


def test_structured_checkpoint_compacts_nested_float_tensors(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(
        {
            "model_state_dict": {"weight": torch.ones(3, dtype=torch.float32)},
            "provenance": {"factorial_mask": "1000000", "seed": 42},
        },
        path,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["model_state_dict"]["weight"].dtype == torch.float16
    assert payload["provenance"] == {"factorial_mask": "1000000", "seed": 42}


def test_queue_state_save_is_valid_atomic_json(tmp_path, monkeypatch):
    state = tmp_path / "queue_state.json"
    monkeypatch.setattr(queue, "QUEUE_FILE", state)
    payload = {"version": 1, "jobs": [{"id": "job", "status": "pending"}]}
    queue.save_queue(payload)
    assert json.loads(state.read_text()) == payload
    assert state.read_text().endswith("\n")
    assert not list(tmp_path.glob(".queue_state.json.*.tmp"))


def test_reconciliation_restores_truncated_ecg_aim_pair():
    jobs = [{"id": "msvae_f_1111114_s42", "status": "pending", "attempts": 0}]
    added = restore_missing_architecture_pairs(jobs)
    assert added == ["ecg_aim_f_1111114_s42"]
    restored = jobs[-1]
    assert restored["status"] == "pending"
    assert "--factorial_mask 1111114" in restored["cmd"]
    assert "--architecture ecg_aim" in restored["cmd"]
