"""CPU-only regression tests. No weights, downloads or package installation required."""
import unittest
import numpy as np
import torch
import boundary_training as bt

class BoundaryTrainingTests(unittest.TestCase):
    def test_crop_uses_only_image_and_background(self):
        config=bt.Config();background=np.zeros((120,160,3),np.uint8)
        image=background.copy();image[45:65,60:100]=180
        crop=bt.find_car_crop(image,background,config)
        self.assertIsNotNone(crop)
        x,y,size=crop
        self.assertLess(x,60);self.assertLess(y,45)
        self.assertGreater(x+size,100);self.assertGreater(y+size,65)
        self.assertIsNone(bt.find_car_crop(background,background,config))

    def test_large_scene_change_is_rejected(self):
        a=np.zeros((100,100,3),np.uint8);b=np.full_like(a,180)
        self.assertIsNone(bt.find_car_crop(b,a,bt.Config()))

    def test_crop_roundtrip_with_padding_and_pixel_centers(self):
        points=np.array([[0.,0.],[11.2,33.9],[90.,50.]])
        crop=(-12,-9,120)
        projected=bt.to_crop(points,crop,384)
        np.testing.assert_allclose(bt.from_crop(projected,crop,384),points,atol=1e-12)
        # Resizing maps source pixel centre x=0 to destination centre (0.5*scale - 0.5).
        np.testing.assert_allclose(bt.to_crop([[0,0]],(0,0,100),200),[[.5,.5]])
        image=np.full((80,100,3),255,np.uint8)
        self.assertEqual(bt.crop_image(image,crop,64).shape,(64,64,3))

    def test_eight_heatmaps_can_overlap_without_losing_labels(self):
        c=bt.Config(image_size=64,heatmap_size=32,sigma_px=1)
        points=torch.tensor([[[20.,20.],[22.,20.],[30.,30.],[32.,30.],[40.,40.],[42.,40.],[15.,45.],[17.,45.]]])
        targets=bt.gaussian_targets(points,c)
        self.assertEqual(tuple(targets.shape),(1,8,32,32))
        torch.testing.assert_close(targets.sum((-2,-1)),torch.ones(1,8))
        decoded,mass=bt.decode_heatmaps(targets.clamp_min(1e-30).log(),c)
        self.assertLess(float(torch.linalg.vector_norm(decoded-points,dim=-1).max()),.2)
        self.assertTrue(bool((mass>.8).all()))

    def test_diffuse_maps_keep_raw_points(self):
        c=bt.Config(image_size=64,heatmap_size=32)
        points,mass=bt.decode_heatmaps(torch.zeros(1,8,32,32),c)
        self.assertTrue(bool(torch.isfinite(points).all()))
        self.assertTrue(bool((mass<c.min_peak_mass).all()))

    def test_whole_tread_rule_and_sampling_review(self):
        # Straight entry, 16 cm ground tread. White-line outer edge is y=-7.
        points=np.tile([[-10.,-7.16],[-10.,-7.0]],(4,1))
        self.assertIsNot(bt.verdict(bt.boundary_interval(points)),True)
        points[:,1]-=.0002
        self.assertIsNone(bt.verdict(bt.boundary_interval(points)))
        points[:,1]-=.001
        self.assertIs(bt.verdict(bt.boundary_interval(points)),True)

    def test_geometry_gates_do_not_fabricate_decisions(self):
        camera={'resolution':[100,100],'ground_plane_homography':[[1,0,20],[0,1,20],[0,0,1]]}
        points=np.tile([[10.,12.8],[10.,12.96]],(4,1))
        config=bt.Config()
        decision,_,reason=bt.decide(points,np.ones(8),camera,config)
        self.assertTrue(decision);self.assertIsNone(reason)
        self.assertEqual(bt.decide(points,np.zeros(8),camera,config)[2],'diffuse_heatmap')
        points[1]=points[0]
        self.assertEqual(bt.decide(points,np.ones(8),camera,config)[2],'implausible_tread_width')

if __name__=='__main__':unittest.main()
