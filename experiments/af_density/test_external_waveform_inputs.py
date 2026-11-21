import unittest
import numpy as np
from external_waveform_inputs import native_window_layout, model_window


class ExternalInputTests(unittest.TestCase):
    def test_native_tails_cover_every_sample_without_changing_clock(self):
        for fs in [128, 200]:
            for remainder in [0, 1, fs-1, 30*fs-1]:
                length = 83+90*fs+remainder
                inputs, starts, ends = native_window_layout(length, 83, fs)
                self.assertTrue((inputs+30*fs <= length).all())
                self.assertEqual(starts[0], 83);self.assertEqual(ends[-1], length)
                covered = np.zeros(length, np.uint8)
                for a, b in zip(starts, ends): covered[a:b] += 1
                self.assertTrue((covered[83:] == 1).all());self.assertFalse(covered[:83].any())

    def test_200Hz_source_values_preserved_bitwise_and_missing_not_imputed(self):
        x = np.random.default_rng(4).normal(size=(6000, 2)).astype(np.float32)
        out, available = model_window(x, 200)
        np.testing.assert_array_equal(out, x);self.assertTrue(available.all())
        x[100, 0] = np.nan;out, available = model_window(x, 200)
        self.assertTrue(np.isnan(out[:, 0]).all());self.assertEqual(available.tolist(), [False, True])
        np.testing.assert_array_equal(out[:, 1], x[:, 1])

    def test_128Hz_impulse_timing_and_exact_input_contract(self):
        x = np.zeros((3840, 2), np.float32);x[1280] = 1
        out, available = model_window(x, 128)
        self.assertEqual(out.shape, (6000, 2));self.assertEqual(int(np.argmax(out[:, 0])), 2000)
        self.assertTrue(available.all())
        with self.assertRaises(ValueError): native_window_layout(1000, 0, 128)
        with self.assertRaises(ValueError): model_window(x[:-1], 128)


if __name__ == '__main__': unittest.main()
