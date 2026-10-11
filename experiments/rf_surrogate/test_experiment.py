"""Independent analytical checks; these do not replace running Sionna validation."""
import unittest
import numpy as np
import experiment as e

class RFMathChecks(unittest.TestCase):
    def test_box_segment(self):
        a=np.array([0.,-1,-1]);b=np.array([2.,1,1])
        self.assertAlmostEqual(e.segment_fraction(np.array([-1.,0,0]),np.array([3.,0,0]),a,b),.5)
        self.assertEqual(e.segment_fraction(np.array([-1.,3,0]),np.array([3.,3,0]),a,b),0)

    def test_two_equal_paths(self):
        y=e.descriptors(np.array([1.,1.]),np.array([1e-9,3e-9]),[np.zeros(2)]*4)
        self.assertAlmostEqual(y[1],10*np.log10(2))
        self.assertAlmostEqual(y[2],.5)
        self.assertAlmostEqual(y[3],2.)
        self.assertAlmostEqual(y[4],1.)
        self.assertAlmostEqual(y[5],2.)

    def test_invalid_paths_excluded(self):
        y=e.descriptors(np.array([1.,0.]),np.array([1e-9,-1]),[np.zeros(2)]*4)
        self.assertEqual(y[0],1)
        self.assertAlmostEqual(y[2],1.)
        self.assertAlmostEqual(y[4],0.)

    def test_angular_wrap(self):
        phi=np.deg2rad([179.,-179.])
        y=e.descriptors(np.ones(2),np.array([1e-9,3e-9]),[np.zeros(2),phi,np.zeros(2),phi])
        self.assertAlmostEqual(y[10],0.)
        self.assertLess(y[11],-.999)
        self.assertLess(y[12],.02)

if __name__=='__main__':unittest.main()
