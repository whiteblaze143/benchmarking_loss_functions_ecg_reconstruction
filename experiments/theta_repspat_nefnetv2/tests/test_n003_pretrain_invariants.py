"""Executable data, model, normalization, and provenance gates for N003a."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from theta_repspat.clinical_dataset import (
    CANONICAL_12LEAD_ANGLES_RAD, DISK_TO_CANONICAL, PTBXLClinicalAnyPairs,
    STANDARD_12LEAD_ANGLES_RAD, normalize_fixed_train_bounds,
)
from theta_repspat.vendor import instantiate_author_model

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/mithunmanivannan/data/ptb_xl/tensors")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def test_pretrain_branch_dynamic_cardinality():
    model = instantiate_author_model(ROOT / "author_code/nefnet_v2", DEVICE, super_mode="pretrain")
    model.eval()
    assert not hasattr(model, "mlp_list")
    assert not hasattr(model, "W_encoder_list")
    assert hasattr(model, "mlp3") and hasattr(model, "W_encoder3")
    for k in (1, 2, 3):
        x = torch.randn(2, 2 + k, 4608, device=DEVICE)
        indices = [0, 1] + list(range(2, 2 + k))
        input_angles = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD[indices]).repeat(2, 1, 1).to(DEVICE)
        query_angles = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD[7]).repeat(2, 1, 1).to(DEVICE)
        with torch.no_grad():
            output = model(x, input_angles, query_angles)
        assert output.shape == (2, 1, 4608)
        assert torch.isfinite(output).all()


def test_pretrain_gradient_flow():
    model = instantiate_author_model(ROOT / "author_code/nefnet_v2", DEVICE, super_mode="pretrain")
    model.train()
    x = torch.randn(2, 4, 4608, device=DEVICE, requires_grad=True)
    input_angles = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD[[0, 1, 2, 4]]).repeat(2, 1, 1).to(DEVICE)
    query_angles = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD[7]).repeat(2, 1, 1).to(DEVICE)
    model(x, input_angles, query_angles).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(float(x.grad[:, slot].norm()) > 0 for slot in range(4))


def test_canonical_reordering_and_full_limb_algebra():
    errors = {name: [] for name in ("III", "aVR", "aVL", "aVF")}
    for filepath in sorted((DATA / "train").glob("*.pt"))[:64]:
        disk = torch.load(filepath, map_location="cpu", weights_only=True)
        if isinstance(disk, dict):
            disk = disk["ecg"]
        x = disk[DISK_TO_CANONICAL]
        expected = {"III": x[1] - x[0], "aVR": -(x[0] + x[1]) / 2,
                    "aVL": x[0] - x[1] / 2, "aVF": x[1] - x[0] / 2}
        for name, index in zip(errors, range(8, 12)):
            errors[name].append((x[index] - expected[name]).abs().flatten())
    for name, chunks in errors.items():
        values = torch.cat(chunks)
        assert float(values.mean()) < 5e-4, name
        assert float(torch.quantile(values, 0.99)) < 1.1e-3, name
        assert float(values.max()) < 1.6e-3, name


def test_independent_sampling_and_fixed_normalization():
    dataset = PTBXLClinicalAnyPairs(DATA, "train", lead_cardinality=None, seed=42)
    for epoch in range(32):
        dataset.set_epoch(epoch)
        sample = dataset[epoch]
        observed = sample["input_indices"].tolist()
        query = int(sample["target_index"])
        assert observed[:2] == [0, 1]
        assert 3 <= len(observed) <= 5
        assert set(observed + [query]) <= set(range(8))
        assert query in range(2, 8) and query not in observed
        assert torch.all((sample["input"] > 0) & (sample["input"] < 1))
        assert torch.all((sample["target"] > 0) & (sample["target"] < 1))


def test_fixed_transform_has_zero_sigmoid_support_mismatch():
    values = torch.tensor([-100.0, -4.0, 0.0, 4.0, 100.0])
    normalized = normalize_fixed_train_bounds(values)
    assert torch.all((normalized > 0) & (normalized < 1))


def _manifest(root: Path):
    excluded = {"__pycache__"}
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or excluded.intersection(path.parts) or path.suffix == ".pyc":
            continue
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_vendored_author_tree_matches_reference():
    reference = ROOT.parents[1] / "external/NEFNET-v2-main"
    vendored = ROOT / "author_code/nefnet_v2"
    assert _manifest(vendored) == _manifest(reference)


def test_canonical_angles_follow_reordering():
    np.testing.assert_array_equal(CANONICAL_12LEAD_ANGLES_RAD,
                                  STANDARD_12LEAD_ANGLES_RAD[DISK_TO_CANONICAL])
