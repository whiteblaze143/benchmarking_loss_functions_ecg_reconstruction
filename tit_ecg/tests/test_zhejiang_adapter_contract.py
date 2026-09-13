import numpy as np

from tit_ecg.src.dataset_adapters import ZhejiangAdapter
from tit_ecg.scripts.run_empirical_multi_dataset_benchmark import beat_ids_from_segmentation


def test_zhejiang_native_signal_and_mask_share_the_500hz_coordinate_system():
    record = ZhejiangAdapter().load_record("1010096", n_samples=2000)

    assert record["fs"] == 500.0
    assert record["source_sampling_rate"] == 2000.0
    assert record["ecg"].shape == (2000, 12)
    assert record["segmentation"].shape == (2000,)
    assert record["preprocessing"] == "polyphase_2000_to_500_hz_and_label_stride_4"

    beat_ids = beat_ids_from_segmentation(record["segmentation"])
    qrs_beat_ids = beat_ids[record["segmentation"] == 2]
    assert len(np.unique(qrs_beat_ids)) >= 5
