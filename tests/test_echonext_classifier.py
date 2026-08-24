from pathlib import Path

import numpy as np

from scripts.echonext_classifier import (
    SHD_LABEL_COLUMNS,
    SHD_TASKS,
    load_echonext_test_metadata,
)


def test_echonext_task_and_label_contracts_are_one_to_one():
    assert len(SHD_TASKS) == len(SHD_LABEL_COLUMNS) == 12
    assert len(set(SHD_TASKS)) == 12
    assert SHD_TASKS[-1] == "shd_moderate_or_greater"
    assert SHD_LABEL_COLUMNS[-1] == "shd_moderate_or_greater_flag"


def test_echonext_official_metadata_alignment():
    root = Path(__file__).parents[1]
    official = (
        root
        / "ecg_fm_integration"
        / "echonext_minimodel_repo"
        / "7-EchoNext Minimodel"
    )
    metadata, tabular, labels = load_echonext_test_metadata(
        root / "data/echonext/echonext_metadata_100k.csv",
        official
        / "models/echonext_multilabel_minimodel/tabular_transformer.joblib",
    )
    assert len(metadata) == 5442
    assert tabular.shape == (5442, 7)
    assert labels.shape == (5442, 12)
    assert np.isfinite(tabular).all()
    assert np.isin(labels, [0.0, 1.0]).all()
