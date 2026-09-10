import json
import torch

from scripts.train_ecg_aim_lvcg_embedding import BeatBoundaryDataset, collate, epoch
from unified_latents.engineering.models.ecg_aim_lvcg_variant import LVCGStyleLeadIEmbedding


def test_collate_and_one_training_epoch():
    batch = [
        (torch.randn(1, 128), torch.tensor([[0, 64], [64, 128]])),
        (torch.randn(1, 128), torch.tensor([[0, 128]])),
    ]
    lead_i, bounds, mask = collate(batch)
    assert lead_i.shape == (2, 1, 128)
    assert bounds.shape == (2, 2, 2)
    assert mask.tolist() == [[True, True], [True, False]]
    model = LVCGStyleLeadIEmbedding(
        beat_len=32, state_dim=16, rhythm_dim=8, max_beats=4,
        stem_channels=8, stage_channels=(8, 8), stage_blocks=(1, 1), dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = epoch(model, [(lead_i, bounds, mask)], torch.device("cpu"), optimizer)
    assert metrics["records"] == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_uniform_control_preserves_count_not_boundaries(tmp_path):
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    torch.save(torch.randn(12, 5000), tensor_dir / "7.pt")
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"record_id": "7", "beat_bounds": [[0, 100], [100, 300], [300, 5000]]}) + "\n")
    dataset = BeatBoundaryDataset(tensor_dir, index, boundary_control="uniform")
    _, bounds = dataset[0]
    assert bounds.shape == (3, 2)
    assert bounds.tolist() != [[0, 100], [100, 300], [300, 5000]]
    assert bounds[0, 0] == 0 and bounds[-1, 1] == 5000
