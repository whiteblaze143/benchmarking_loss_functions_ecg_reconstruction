import unittest
import numpy as np
import torch
from raw_waveform_model import RawMaskResNet, availability_ids


class RawWaveformTests(unittest.TestCase):
    def test_each_window_receives_all_three_patterns_per_cycle(self):
        ids = np.arange(20)
        schedule = np.stack([availability_ids(ids, epoch) for epoch in range(3)])
        np.testing.assert_array_equal(np.sort(schedule, axis=0), np.tile(np.arange(3)[:, None], (1, 20)))

    def test_zero_observed_channels_are_rejected(self):
        model = RawMaskResNet()
        with self.assertRaisesRegex(ValueError, 'No observed channel'):
            model(torch.zeros(1, 2, 6000), torch.zeros(1, 2, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
