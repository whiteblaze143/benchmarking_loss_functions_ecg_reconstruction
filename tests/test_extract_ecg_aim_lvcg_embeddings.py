import numpy as np
import torch

from scripts.extract_ecg_aim_lvcg_embeddings import extract
from scripts.train_ecg_aim_lvcg_embedding import collate
from unified_latents.engineering.models.ecg_aim_lvcg_variant import LVCGStyleLeadIEmbedding


def test_extract_preserves_component_order_and_count():
    model = LVCGStyleLeadIEmbedding(
        beat_len=32, state_dim=16, rhythm_dim=8, max_beats=4,
        stem_channels=8, stage_channels=(8, 8), stage_blocks=(1, 1), dropout=0.0,
    )
    batch = collate([
        (torch.randn(1, 128), torch.tensor([[0, 64], [64, 128]])),
        (torch.randn(1, 128), torch.tensor([[0, 128]])),
    ])
    arrays = extract(model, [batch], torch.device("cpu"))
    assert arrays["ecg_emb"].shape == (2, 40)
    np.testing.assert_allclose(
        arrays["ecg_emb"],
        np.concatenate((arrays["emb_struct"], arrays["emb_dynamic"], arrays["emb_rhythm"]), axis=1),
    )
