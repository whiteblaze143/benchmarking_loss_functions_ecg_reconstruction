import unittest
import numpy as np
from horizon_inputs import contiguous_prefix


class HorizonTests(unittest.TestCase):
    def test_prefix_never_uses_future_windows_or_changes_native_clock(self):
        for fs in [128, 200]:
            a = np.arange(17, 17+12*3600*fs, 30*fs, dtype=np.int64);b = a+30*fs
            r = contiguous_prefix(a, a, b, fs, 6)
            self.assertEqual(len(r['indices']), 720)
            self.assertEqual(r['source_start_sample'], 17)
            self.assertEqual(r['source_end_sample'], 17+6*3600*fs)
            self.assertEqual(contiguous_prefix(a, a, b, fs, 24)['status'], 'ineligible_insufficient_contiguous_time')

    def test_input_lookahead_is_rejected(self):
        a = np.arange(0, 6*3600*200, 6000);b = a+6000;inputs = a.copy();inputs[-1] += 1
        with self.assertRaises(ValueError): contiguous_prefix(inputs, a, b, 200, 6)


if __name__ == '__main__': unittest.main()
