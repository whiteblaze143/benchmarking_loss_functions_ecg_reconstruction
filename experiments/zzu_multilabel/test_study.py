"""Guard multilabel semantics, nominal operators and tied-score bootstrap metrics."""
import unittest
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from study import targets,Q,QN
from analyze import weighted_ap,weighted_auc,ranking
class Contracts(unittest.TestCase):
 def test_normal_and_overlapping_labels(self):
  f=pd.DataFrame({'labels_json':['["Normal"]','["Acute Myocardial Infarction", "Severe Hyperkalemia"]']});y=targets(f)
  self.assertEqual(y.shape,(2,15));self.assertEqual(y[0].sum(),0);self.assertEqual(y[1].sum(),2);self.assertEqual(y[1,0],1);self.assertEqual(y[1,11],1)
 def test_nominal_operator_and_polarity(self):
  basis=np.random.default_rng(42).normal(size=(256,8));nominal=basis@(Q*QN[:,None]).T
  np.testing.assert_allclose(basis@Q.T,nominal/QN);np.testing.assert_array_equal(Q[1],np.eye(8)[1]);np.testing.assert_allclose((-basis)@(-Q).T,basis@Q.T)
 def test_weighted_bootstrap_matches_sklearn_with_ties(self):
  y=np.array([0,1,0,1,1,0]);s=np.array([.1,.4,.4,.7,.7,.9]);w=np.array([[1,2,1,1,0,2],[2,0,1,3,2,1]],float)
  for i,weight in enumerate(w):
   self.assertAlmostEqual(weighted_ap(y,s,w)[i],average_precision_score(y,s,sample_weight=weight));self.assertAlmostEqual(weighted_auc(y,s,w)[i],roc_auc_score(y,s,sample_weight=weight))
 def test_no_positive_is_undefined(self):
  y=np.zeros(4);s=np.arange(4)/4;self.assertIsNone(ranking(y,s)['auprc']);self.assertIsNone(ranking(y,s)['auroc']);self.assertTrue(np.isnan(weighted_ap(y,s,np.ones((1,4))))[0]);self.assertTrue(np.isnan(weighted_auc(y,s,np.ones((1,4))))[0])
if __name__=='__main__':unittest.main()
