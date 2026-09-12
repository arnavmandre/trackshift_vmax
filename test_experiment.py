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
    def test_perfect_simulator_contacts(self):
        from simulator.generate import make_specs,trajectory
        import geometry as g
        spec=make_specs(1,1207,12,2,480,270)[0]
        g.W,g.H=480,270
        cam=g.camera(spec['id'],spec['camera_position'],spec['camera_target'],spec['fov'])
        rows,_=trajectory(spec)
        raw=[]
        for f in rows:
            if f['car_id']!='car_1':continue
            uv=transform(f['contacts_world'],cam['ground_plane_homography'])
            raw.append({'frame':f['frame_idx'],'detections':[{'score':1,'keypoints':np.c_[uv,np.ones(4)].tolist(),'box':[0,0,480,270]}]})
        tracks=observations({'calibration':{'homography':cam['ground_plane_homography']}},raw,.3)
        output={f['frame']:f['margin_m'] for t in tracks for f in t}
        for f in rows:
            if f['car_id']=='car_1':self.assertAlmostEqual(output[f['frame_idx']],f['footprint_margin_m'],places=8)

if __name__=='__main__':unittest.main()
