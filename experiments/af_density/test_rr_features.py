import unittest
import numpy as np
from rr_features import features,layout,NAMES
class RRTests(unittest.TestCase):
    def test_regular(self):
        x,valid=features(np.full((1,60),1000.))
        self.assertTrue(valid[0]);d=dict(zip(NAMES,x[0]))
        self.assertEqual(d['mean_rr_ms'],1000);self.assertEqual(d['rmssd_ms'],0);self.assertEqual(d['hist_entropy'],0)
    def test_alternation(self):
        x,_=features(np.tile([900.,1100.],30)[None,:]);d=dict(zip(NAMES,x[0]))
        self.assertEqual(d['rmssd_ms'],200);self.assertEqual(d['pnn50'],1);self.assertEqual(d['turning_point_fraction'],1)
    def test_does_not_join_invalid_gaps(self):
        x=np.zeros((1,60));x[0,[0,2,4,6]]=1000
        f,valid=features(x);self.assertFalse(valid[0]);self.assertTrue(np.isnan(f).all())
    def test_tail_coverage_without_padding(self):
        for length in range(91,300):
            starts,lo,hi=layout(length,31)
            self.assertTrue((starts>=31).all());self.assertTrue((starts+60<=length).all())
            self.assertEqual(lo[0],31);self.assertEqual(hi[-1],length)
            np.testing.assert_array_equal(hi[:-1],lo[1:])
    def test_invalid_contract(self):
        with self.assertRaises(ValueError):features(np.ones((1,59)))
        with self.assertRaises(ValueError):layout(90,31)
if __name__=='__main__':unittest.main()
