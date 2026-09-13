import numpy as np
import pytest

pytest.importorskip("anndata")

from theta_repspat.repspat_adapter import panorama_to_anndata


def test_temporal_domain_and_attribute_shape():
    panorama = np.arange(40, dtype=np.float32).reshape(8, 5)
    adata = panorama_to_anndata(panorama, sampling_hz=250.0)

    assert adata.shape == (5, 8)
    np.testing.assert_array_equal(adata.X, panorama.T)
    np.testing.assert_allclose(adata.obsm["spatial"][:, 0], np.arange(5) / 250.0)
    np.testing.assert_array_equal(adata.obsm["spatial"][:, 1], 0.0)
    assert adata.uns["theta_repspat"]["input_leads"] == ["I", "II", "V3"]
