import unittest
from unittest.mock import patch
import numpy as np
from observed_waveform import extract_channel,rr_statistics
from rr_features import features
from waveform_protocol import window_layout,count_overlap

class ObservedWaveformTests(unittest.TestCase):
    def test_variable_rr_matches_native_statistics(self):
        rng=np.random.default_rng(42)
        for _ in range(20):
            rr=rng.integers(150,4501,60).astype(float)
            expected,ok=features(rr[None,:]);actual,valid=rr_statistics(rr)
            self.assertEqual(valid,bool(ok[0]));np.testing.assert_allclose(actual,expected[0],rtol=1e-6,atol=1e-5,equal_nan=True)
    def test_variable_count_not_padded(self):
        f,valid=rr_statistics([1000]*5)
        self.assertTrue(valid);self.assertEqual(f[0],1);self.assertEqual(f[2],1000)
    def test_invalid_rr_adjacency_not_joined(self):
        f,valid=rr_statistics([1000,0,1000,0,1000,0,1000])
        self.assertFalse(valid);self.assertTrue(np.isnan(f).all())
    def test_no_manufactured_beats_after_failure(self):
        with patch('observed_waveform.xqrs_detect',side_effect=ValueError('fixture')):
            r=extract_channel(np.arange(6000,dtype=float))
        self.assertEqual(r['status'],3);self.assertEqual(len(r['peaks']),0);self.assertFalse(r['phase_valid']);self.assertFalse(r['rr_valid'])
    def test_constant_and_nonfinite_are_explicit(self):
        self.assertEqual(extract_channel(np.zeros(6000))['status'],1)
        x=np.zeros(6000);x[5]=np.nan;self.assertEqual(extract_channel(x)['status'],2)
    def test_single_channel_shape_only(self):
        with self.assertRaises(ValueError):extract_channel(np.zeros((6000,2)))
    def test_authentic_inclusive_rr_limits(self):
        with patch('observed_waveform.xqrs_detect',return_value=np.array([10,70,470,529,930])):
            r=extract_channel(np.arange(6000,dtype=float))
        self.assertEqual(r['accepted_intervals'],2);self.assertTrue(r['phase_valid']);self.assertAlmostEqual(r['represented_fraction'],460/6000)
    def test_layout_covers_tail_exactly(self):
        for length in [12000,12001,17999,18000,18001,123456]:
            start,lo,hi=window_layout(length,6000)
            self.assertEqual(lo[0],6000);self.assertEqual(hi[-1],length)
            np.testing.assert_array_equal(hi[:-1],lo[1:]);self.assertTrue((start>=6000).all());self.assertTrue((start+6000<=length).all())
    def test_interval_labels_use_half_open_sample_ranges(self):
        np.testing.assert_array_equal(count_overlap(np.array([0,10,20]),np.array([10,20,30]),[(5,10),(20,25)]),[5,0,5])

if __name__=='__main__':unittest.main()
