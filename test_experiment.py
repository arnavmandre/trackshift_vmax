"""Small correctness checks before spending training compute."""
import unittest
import numpy as np
from experiment import transform,groups,overlap,observations

class Checks(unittest.TestCase):
    def test_projection_roundtrip(self):
        H=np.array([[20,3,100],[1,15,70],[.01,.02,1.]])
        p=np.array([[0,0],[5,2],[-1,8]])
        np.testing.assert_allclose(transform(transform(p,H),np.linalg.inv(H)),p,atol=1e-10)
    def test_events_do_not_bridge_legal_or_missing_frames(self):
        self.assertEqual(groups([1,2,4,7]),[(1,2),(4,4),(7,7)])
        self.assertAlmostEqual(overlap((1,3),(2,4)),.5)
    def test_perfect_simulator_contacts_on_flat_ground(self):
        # Off the kerb (elevation 0), a perfect detector still round-trips
        # through the flat-ground homography to the exact ground-truth margin.
        from simulator.generate import make_specs,ring_cameras,trajectory
        import geometry as g
        spec=make_specs(1,1207,12,2,480,270,angles=6)[0]
        g.W,g.H=480,270
        cam=ring_cameras(spec['id'],spec['incident_centre'],spec['angles'],spec['ring_radius'],spec['ring_height'],spec['ring_fov'])[0]
        rows,_=trajectory(spec)
        raw=[]
        for f in rows:
            if f['car_id']!='car_1':continue
            uv,_=g.project(np.asarray(f['contacts_world']),cam)
            raw.append({'frame':f['frame_idx'],'detections':[{'score':1,'keypoints':np.c_[uv,np.ones(4)].tolist(),'box':[0,0,480,270]}]})
        tracks=observations({'calibration':{'homography':cam['ground_plane_homography']}},raw,.3)
        output={f['frame']:f['margin_m'] for t in tracks for f in t}
        checked_a_flat_frame=False
        for f in rows:
            if f['car_id']!='car_1':continue
            if max(p[2] for p in f['contacts_world'])>0:continue  # skip kerb frames
            checked_a_flat_frame=True
            self.assertAlmostEqual(output[f['frame_idx']],f['footprint_margin_m'],places=8)
        self.assertTrue(checked_a_flat_frame,'spec/seed produced no flat-ground frames to check')

    def test_kerb_elevation_causes_small_bounded_backprojection_bias(self):
        # On the kerb, a raised contact point projects to where it actually
        # appears; backprojecting that pixel through the flat-ground homography
        # (a real single-camera limitation, not a bug) recovers a slightly
        # wrong position. The bias should be real but small.
        from simulator.generate import make_specs,ring_cameras,trajectory
        import geometry as g
        spec=make_specs(1,1207,12,2,480,270,angles=6)[0]
        g.W,g.H=480,270
        cam=ring_cameras(spec['id'],spec['incident_centre'],spec['angles'],spec['ring_radius'],spec['ring_height'],spec['ring_fov'])[0]
        rows,_=trajectory(spec)
        raw=[]
        for f in rows:
            if f['car_id']!='car_1':continue
            uv,_=g.project(np.asarray(f['contacts_world']),cam)
            raw.append({'frame':f['frame_idx'],'detections':[{'score':1,'keypoints':np.c_[uv,np.ones(4)].tolist(),'box':[0,0,480,270]}]})
        tracks=observations({'calibration':{'homography':cam['ground_plane_homography']}},raw,.3)
        output={f['frame']:f['margin_m'] for t in tracks for f in t}
        deviations=[abs(output[f['frame_idx']]-f['footprint_margin_m']) for f in rows
                    if f['car_id']=='car_1' and max(p[2] for p in f['contacts_world'])>0 and f['frame_idx'] in output]
        self.assertTrue(deviations,'spec/seed produced no kerb frames to check')
        self.assertGreater(max(deviations),1e-6,'elevation should visibly perturb the flat backprojection')
        self.assertLess(max(deviations),.15,'kerb parallax bias should stay small relative to the ~5cm kerb height')

if __name__=='__main__':unittest.main()
