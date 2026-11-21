import unittest
import numpy as np,pandas as pd
from protocol import locked_threshold,metrics,unit_normalize_windows,candidates

class ProtocolTests(unittest.TestCase):
    def test_threshold_conservative_ties(self):
        y=np.array([0]*20+[1]*3);s=np.array([.1]*18+[.8,.8]+[.5,.8,.9])
        t=locked_threshold(y,s,.95)
        self.assertGreater(t,.8);self.assertGreaterEqual(np.mean(s[y==0]<t),.95)
    def test_failed_positive_is_missed(self):
        m=metrics([1,1,0],[.9,np.nan,.9],['completed','failed','completed'],.5)
        self.assertEqual(m['sensitivity_all_eligible'],.5);self.assertEqual(m['positive_coverage'],.5)
        self.assertEqual(m['inappropriate_positive_fraction_all_controls'],1)
    def test_near_copy_retrieval(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(12,12,500)).astype('float32');x[1]=3*x[0]+2
        z=unit_normalize_windows(x);p=z.reshape(12,12,50,10).mean(-1).reshape(12,600);p/=np.linalg.norm(p,axis=1,keepdims=True)
        self.assertIn((0,1),map(tuple,candidates(p)));self.assertGreater(float(z[0]@z[1]),.9999)
    def test_paired_bootstrap_preserves_identity(self):
        from evaluate import paired_group_intervals
        f=pd.DataFrame({'record_uid':['a','b','c','d'],'record_group':['g1','g1','g2','g2'],'label':[1,0,1,0],'score':[.9,.1,.8,.2],'decision_status':['completed']*4,'threshold':[.5]*4})
        ci=paired_group_intervals(f,f.copy(),replicates=20)
        self.assertEqual(ci['sensitivity_all_eligible']['low'],0)
        self.assertEqual(ci['auroc']['high'],0)
        with self.assertRaises(ValueError):paired_group_intervals(f,f.iloc[::-1],replicates=2)
    def test_shape_is_not_record_independence(self):
        from protocol import CONFIGS
        self.assertEqual(set(CONFIGS),{'full12','I','II','V1','I_II','II_V1','limb6','precordial6'})
        self.assertEqual(CONFIGS['II_V1'],['II','V1'])
if __name__=='__main__':unittest.main()
