"""Run: python -m unittest discover -s spam/vmax -p 'test_*.py'"""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
import tyre_boundaries as tb


class BoundaryTests(unittest.TestCase):
    def row(self,y=-7.04,heading=0,steer=0):
        return dict(frame_idx=0,t=0.,car_id='car_1',heading_rad=heading,visual_steering_rad=steer,
                    contact_points_world={k:[-10.,y] for k in tb.g.KEYS},
                    corner_excess_m={k:abs(y)-7 for k in tb.g.KEYS},
                    min_excess_m=abs(y)-7,max_excess_m=abs(y)-7,is_violation=abs(y)>7)

    def test_center_out_but_tread_still_in(self):
        row=self.row();result=tb.annotate_row(row)
        self.assertTrue(row['is_violation'])
        self.assertFalse(result['is_violation'])
        self.assertTrue(result['legacy_center_rule']['is_violation'])
        self.assertNotIn('min_excess_m',result)

    def test_full_tread_out_and_borderline(self):
        self.assertTrue(tb.annotate_row(self.row(-7.09))['is_violation'])
        self.assertIsNot(tb.annotate_row(self.row(-7.08))['is_violation'],True)
        self.assertIsNone(tb.annotate_row(self.row(-7.0802))['is_violation'])

    def test_actual_profile_width_and_height(self):
        b=tb.wheel_boundary(self.row(),'front_left',None)
        outline=np.array(b['lower_outline_world_xyz']);ends=np.array(b['ground_tread_endpoints_world_xyz'])
        self.assertAlmostEqual(np.linalg.norm(ends[1]-ends[0]),.16)
        self.assertAlmostEqual(outline[0,2],.09)
        self.assertAlmostEqual(np.linalg.norm(outline[0,:2]-outline[-1,:2]),.36)
        np.testing.assert_allclose(ends[:,2],0)

    def test_front_steering_and_rear_orientation(self):
        row=self.row(heading=.3,steer=.4)
        front=tb.wheel_boundary(row,'front_left',None);rear=tb.wheel_boundary(row,'rear_left',None)
        for item,angle in [(front,.7),(rear,.3)]:
            ends=np.array(item['ground_tread_endpoints_world_xyz'])
            np.testing.assert_allclose((ends[1]-ends[0])[:2],.16*np.array([-np.sin(angle),np.cos(angle)]))

    def test_interval_contains_dense_ground_truth(self):
        rng=np.random.default_rng(14)
        for _ in range(100):
            row=self.row(heading=float(rng.uniform(-3,3)),steer=.2)
            point=rng.uniform([-20,-10],[50,65])
            row['contact_points_world']={k:point.tolist() for k in tb.g.KEYS}
            b=tb.wheel_boundary(row,'front_left',None)
            ends=np.array(b['ground_tread_endpoints_world_xyz'])[:,:2]
            dense=ends[0]+np.linspace(0,1,20001)[:,None]*(ends[1]-ends[0])
            expected=float((tb.g.distance(dense)-7).min())
            lo,hi=b['minimum_excess_m_interval']
            self.assertLessEqual(lo-1e-10,expected)
            self.assertLessEqual(expected,hi+1e-10)

    def test_projection_matches_ground_homography(self):
        camera=tb.g.camera('test',[10,-20,8],[-10,-7,0],40)
        b=tb.wheel_boundary(self.row(),'front_left',camera)
        world=np.array(b['ground_tread_endpoints_world_xyz'])
        h=np.column_stack([world[:,:2],np.ones(2)])@np.array(camera['ground_plane_homography']).T
        np.testing.assert_allclose(h[:,:2]/h[:,2,None],b['ground_tread_endpoints_image']['pixels'])

    def test_source_unchanged_and_repeat_safe(self):
        row=self.row();before=copy.deepcopy(row)
        one=tb.annotate_row(row);two=tb.annotate_row(one)
        self.assertEqual(row,before);self.assertEqual(one,two)
        json.dumps(two,allow_nan=False)

    def test_events_actual_frames_and_fps(self):
        rows=[]
        for i,y in enumerate([-7.04,-7.09,-7.09,-7.0802,-7.04]):
            row=self.row(y);row.update(frame_idx=i,t=i/60);rows.append(tb.annotate_row(row))
        summary=tb.event_summary(rows,60)['car_1']
        self.assertEqual(summary['violation_frames'],2);self.assertEqual(summary['review_frames'],1)
        self.assertEqual(summary['events'][0]['start_frame'],1)
        self.assertAlmostEqual(summary['events'][0]['end_time_exclusive_s'],3/60)


if __name__=='__main__': unittest.main()
