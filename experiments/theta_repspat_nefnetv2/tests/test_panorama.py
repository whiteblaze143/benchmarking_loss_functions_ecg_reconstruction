import numpy as np
import pytest
import torch

from theta_repspat.panorama import (
    CANONICAL_ANGLES_RAD,
    INPUT_INDICES,
    PREDICTED_INDICES,
    build_canonical_panorama,
    global_minmax_normalize,
)


class QueryWitness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x, input_thetas, query_theta):
        del input_thetas
        value = query_theta[:, :1, None] + query_theta[:, 1:, None]
        return value.expand(x.shape[0], 1, x.shape[-1])


def test_panorama_preserves_observed_leads_and_queries_missing_leads():
    model = QueryWitness().eval()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    result = build_canonical_panorama(model, x)

    assert result.shape == (1, 8, 8)
    torch.testing.assert_close(result[:, list(INPUT_INDICES)], x)
    for index in PREDICTED_INDICES:
        expected = float(CANONICAL_ANGLES_RAD[index].sum())
        torch.testing.assert_close(result[0, index], torch.full((8,), expected))


def test_panorama_rejects_trainable_model():
    model = QueryWitness().eval()
    model.anchor.requires_grad_(True)
    with pytest.raises(ValueError, match="must be frozen"):
        build_canonical_panorama(model, torch.zeros(1, 3, 8))


def test_global_normalization_matches_author_formula():
    ecg = np.arange(16, dtype=np.float32).reshape(8, 2)
    normalized = global_minmax_normalize(ecg)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    np.testing.assert_allclose(normalized, ecg / 15.0)


def test_global_normalization_rejects_constant_record():
    with pytest.raises(ValueError, match="constant ECG"):
        global_minmax_normalize(np.ones((8, 10), dtype=np.float32))
