import unittest
from pathlib import Path
import numpy as np
from rf_math import *

class RFTests(unittest.TestCase):
    def test_projection_roundtrip(self):
        pixels=direction_pixel(direction_grid()).reshape(HEIGHT,WIDTH,2)
        xx,yy=np.meshgrid(np.arange(WIDTH),np.arange(HEIGHT))
        np.testing.assert_allclose(pixels,np.stack([xx,yy],-1),atol=1e-9)

    def test_beamforming_sign_and_coherent_cancellation(self):
        # Noncoplanar test array removes the planar-array mirror ambiguity.
        offsets=np.random.default_rng(7).uniform(-.2,.2,(32,3))
        a=steering_matrix(offsets,.1)
        idx=42*WIDTH+210
        p,_=spectrum_from_channel(a[idx],a)
        self.assertEqual(p.argmax(),idx)
        # Coherent equal-and-opposite paths must cancel, not add as powers.
        with self.assertRaises(ValueError):
            spectrum_from_channel(a[idx]-a[idx],a)

    def test_room_and_split(self):
        root=Path(__file__).parent/'scene_fullroom'/'meshes'
        bounds=mesh_bounds(root)
        rx,x,s=sample_positions(bounds)
        self.assertEqual(x.shape,(600,3))
        self.assertTrue(all(clear_position(p,bounds) for p in x))
        sets=[set(v) for v in s.values()]
        self.assertEqual([len(v) for v in sets],[420,90,90])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        d=np.linalg.norm(x[:,None]-x[None,:],axis=-1)+np.eye(len(x))*100
        self.assertGreaterEqual(d.min(),.14999)
        rx2,x2,s2=sample_positions(bounds)
        np.testing.assert_array_equal(x,x2)

if __name__=='__main__': unittest.main()
