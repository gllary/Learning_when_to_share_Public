import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from hpsafemoe.stage2_calibration import jarvis,matbench
from hpsafemoe.stage2_calibration.none import apply

class ContractTests(unittest.TestCase):
    def test_fixed_grid(self):
        for module in (jarvis,matbench):self.assertEqual(tuple(module.ALPHAS),(0.,.25,.5,.75,1.))
    def test_fallback(self):
        a=np.array([.1,.2,.7],dtype=np.float64);r=np.array([.2,.3,.5])
        for module in (jarvis,matbench):np.testing.assert_array_equal(module.apply_beta(a,r,0),a)
    def test_no_control(self):
        a=np.zeros(3);r=np.array([1.,2.,3.]);np.testing.assert_array_equal(apply(a,r),r)
    def test_liberal_boundary(self):
        self.assertTrue(matbench.accept_liberal(.0001,-1.))
        self.assertFalse(matbench.accept_liberal(0.,0.))
        self.assertFalse(matbench.accept_liberal(1.,-1.00001))
    def test_auc_uses_continuous_scores(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.4,.3,.9])
        self.assertEqual(matbench.score(y,p,True),.75)
if __name__=='__main__':unittest.main()
