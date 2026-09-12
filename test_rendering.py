"""Rendering regression checks: camera scaling, clipping and antialiasing."""
import unittest
import numpy as np
import geometry as g

class Rendering(unittest.TestCase):
    def test_camera_resolution_controls_buffers(self):
        cam=g.camera('test',[0,-8,4],[0,0,0],55)
        cam['resolution']=[96,64]
        rgb,depth=g.render([],cam)
        self.assertEqual(rgb.shape,(64,96,3))
        self.assertEqual(depth.shape,(64,96))

    def test_supersampled_projection_preserves_labels(self):
        cam=g.camera('test',[0,-8,4],[0,0,0],55)
        points=np.array([[0,0,0],[1,.5,.05],[-1,-.5,.025]])
        high=dict(cam,K=(np.diag([2,2,1])@np.asarray(cam['K'])).tolist())
        uv,z=g.project(points,cam);uv_high,z_high=g.project(points,high)
        np.testing.assert_allclose(uv_high/2,uv)
        np.testing.assert_allclose(z_high,z)

    def test_compiled_matches_numpy_with_kerbs_and_near_clipping(self):
        if g.rasterize is None:self.skipTest('optional numba unavailable')
        cam=g.camera('test',[20,-12,8],[20,8,0],55)
        mesh=g.track_mesh()+g.car_mesh(wheel_dz=(.05,0,.02,.03))
        compiled=g.render(mesh,cam)
        kernel=g.rasterize
        try:
            g.rasterize=None
            reference=g.render(mesh,cam)
        finally:g.rasterize=kernel
        np.testing.assert_array_equal(compiled[0],reference[0])
        np.testing.assert_allclose(compiled[1],reference[1])

    def test_compiled_matches_numpy_for_near_plane_crossings(self):
        if g.rasterize is None:self.skipTest('optional numba unavailable')
        cam={'resolution':[64,64],'K':[[32,0,32],[0,32,32],[0,0,1]],
             'position_m':[0,0,0],'R_world_to_camera':np.eye(3).tolist()}
        rng=np.random.default_rng(123)
        mesh=[(rng.uniform(-1,1,(3,3)),tuple(rng.integers(0,256,3))) for _ in range(30)]
        compiled=g.render(mesh,cam);kernel=g.rasterize
        try:
            g.rasterize=None;reference=g.render(mesh,cam)
        finally:g.rasterize=kernel
        np.testing.assert_array_equal(compiled[0],reference[0])
        np.testing.assert_allclose(compiled[1],reference[1])

    def test_near_clipping_and_depth_occlusion(self):
        cam={'resolution':[32,32],'K':[[16,0,16],[0,16,16],[0,0,1]],
             'position_m':[0,0,0],'R_world_to_camera':np.eye(3).tolist()}
        tri=np.array([[-1,-1,1],[1,-1,1],[0,1,1.]])
        near=(tri,(255,0,0));far=(tri*2,(0,255,0))
        first=g.render([near,far],cam);second=g.render([far,near],cam)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertEqual(first[1][16,16],1)
        clipped=tri.copy();clipped[0,2]=.1
        rgb,depth=g.render([(clipped,(255,0,0))],cam)
        self.assertTrue(np.isfinite(depth).any())
        self.assertGreaterEqual(depth.min(),.2)

if __name__=='__main__':unittest.main()
