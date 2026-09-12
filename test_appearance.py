import unittest
import numpy as np
import geometry as g
from simulator.appearance import surface, shadows


class Appearance(unittest.TestCase):
    def test_materials_and_shadows_preserve_depth_and_input(self):
        cam = g.camera('test', [20,-12,8], [20,0,0], 55)
        cam['resolution'] = [96,64]
        cam['K'] = [[70,0,48],[0,70,32],[0,0,1]]
        base = g.render(g.track_mesh(),cam)
        original = base[0].copy()
        enhanced, world, valid = surface(base,cam)
        again, _, _ = surface(base,cam)
        np.testing.assert_array_equal(enhanced[0],again[0])
        result = shadows(enhanced,world,valid,[dict(world_position=[20,-4],heading_rad=0)])
        np.testing.assert_array_equal(result[1],base[1])
        np.testing.assert_array_equal(base[0],original)
        self.assertTrue(np.all(result[0] <= enhanced[0]))
        self.assertFalse(np.array_equal(enhanced[0],original))
