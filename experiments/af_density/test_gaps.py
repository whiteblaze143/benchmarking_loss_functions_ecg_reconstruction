import unittest
import numpy as np
from gaps import apply_gap_strategy

class GapTests(unittest.TestCase):
    def test_context_equal_neighbors(self):
        m=np.array([False,True,True,False])
        for state in [0,1]:
            x=np.array([state,1-state,1-state,state])
            out=apply_gap_strategy(x,x,m,'context')
            np.testing.assert_array_equal(out['predicted'],[state]*4)
            np.testing.assert_array_equal(out['reference'],x)
            self.assertEqual(out['clock'],'original_bins')
    def test_context_different_neighbors(self):
        x=np.array([0,0,0,0,0,1]);m=np.array([False,True,True,True,True,False])
        np.testing.assert_array_equal(apply_gap_strategy(x,x,m,'context')['predicted'],[0,0,0,1,1,1])
    def test_odd_midpoint(self):
        x=np.array([0,0,0,0,1]);m=np.array([False,True,True,True,False])
        np.testing.assert_array_equal(apply_gap_strategy(x,x,m,'context')['predicted'],[0,0,1,1,1])
    def test_nonaf_keeps_reference(self):
        x=np.ones(4,dtype=int);m=np.array([False,True,True,False])
        out=apply_gap_strategy(x,x,m,'nonaf')
        np.testing.assert_array_equal(out['reference'],x)
        np.testing.assert_array_equal(out['predicted'],[1,0,0,1])
    def test_observed_only_explicit_clock_change(self):
        x=np.array([1,0,0,1]);m=np.array([False,True,True,False])
        out=apply_gap_strategy(x,x,m,'observed_only')
        np.testing.assert_array_equal(out['reference'],[1,1])
        self.assertEqual(out['clock'],'compressed_observed_bins')
    def test_terminal_and_all_missing_remain_indeterminate(self):
        x=np.array([0,1]);m=np.array([True,False])
        self.assertEqual(apply_gap_strategy(x,x,m,'context')['status'],'indeterminate_missing_flanking_context')
        self.assertEqual(apply_gap_strategy(x,x,np.ones(2,dtype=bool),'observed_only')['status'],'indeterminate_no_observed_time')
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):apply_gap_strategy([0],[0],[1],'nonaf')
        with self.assertRaises(ValueError):apply_gap_strategy([0],[0],np.array([False]),'silent_impute')
if __name__=='__main__':unittest.main()
