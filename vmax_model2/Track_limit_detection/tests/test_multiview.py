import copy
import unittest
from multiview import fuse_views,fuse_records


def views(votes):
    return [dict(scenario='case',car_id='car_1',frame_idx=20,time_s=20/24,
        camera_name=c,prediction=v,review_reason=None if v is not None else 'no_car_crop',truth=True)
        for c,v in zip(('broadcast','exit','trackside'),votes)]


class MultiViewTests(unittest.TestCase):
    def test_one_reliable_view_can_decide(self):
        r=fuse_views(views([None,False,None]));self.assertIs(r['prediction'],False)
        self.assertEqual(r['supporting_cameras'],['exit'])

    def test_agreement(self):self.assertIs(fuse_views(views([True,True,None]))['prediction'],True)

    def test_conflict_even_with_majority(self):
        r=fuse_views(views([True,True,False]));self.assertIsNone(r['prediction'])
        self.assertEqual(r['review_reason'],'camera_disagreement')

    def test_all_review(self):self.assertEqual(fuse_views(views([None]*3))['review_reason'],'no_reliable_view')

    def test_review_flag_excludes_vote(self):
        v=views([True,False,None]);v[1]['review_reason']='implausible_tread_width'
        self.assertIs(fuse_views(v)['prediction'],True)

    def test_truth_cannot_select_verdict(self):
        v=views([False,False,None]);a=fuse_views(v)
        for r in v:r['truth']=False
        self.assertEqual(a,fuse_views(v))

    def test_synchronization_and_camera_validation(self):
        original=views([True]*3)
        for key,value in [('car_id','car_2'),('scenario','other'),('frame_idx',21),('time_s',1.0),('camera_name','exit')]:
            v=copy.deepcopy(original);v[0][key]=value
            with self.assertRaises(ValueError):fuse_views(v)
        with self.assertRaises(ValueError):fuse_views(original[:2])

    def test_groups_score_once_per_car_frame(self):
        a=views([True,None,True]);b=copy.deepcopy(a)
        for r in b:r['car_id']='car_2'
        self.assertEqual(len(fuse_records(a+b)),2)
        a[0]['truth']=False
        with self.assertRaises(ValueError):fuse_records(a)


if __name__=='__main__':unittest.main()
