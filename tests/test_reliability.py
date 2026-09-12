"""Executable regression tests: python -m unittest discover -s tests -p test_reliability.py."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Row
import unittest
import numpy as np
from vmax_vision import boundary
from vmax_vision.evidence import cache_signature,frame_evidence,sha256
from vmax_vision.footprints import clearance
from vmax_vision.generate import make_specs,trajectory
from vmax_vision.blind import validate_manifest,seal_run,score


def rows(indices,legal=()):
    return [Row(frame_idx=i,time_s=i/24,is_violation=i not in legal,margin_m=.2,score=.8) for i in indices]


class ReliabilityTests(unittest.TestCase):
    def test_dropout_is_counted(self):
        ev=boundary.find_events(rows([0,1,2,20,21,22]),min_frames=1,bridge=4)
        self.assertEqual([(e.start_frame,e.end_frame_inclusive) for e in ev],[(0,2),(20,22)])

    def test_short_missing_gap_can_bridge(self):
        self.assertEqual(len(boundary.find_events(rows([0,1,4,5]),min_frames=1,bridge=2)),1)
        self.assertEqual(len(boundary.find_events(rows([0,1,4,5]),min_frames=1,bridge=1)),2)

    def test_observed_legal_frame_splits(self):
        self.assertEqual(len(boundary.find_events(rows(range(7),legal=[3]),min_frames=1,bridge=4)),2)

    def test_brief_excursion_retained_when_requested(self):
        self.assertEqual(len(boundary.find_events(rows([0]),min_frames=1)),1)

    def test_missing_evidence_reduces_score(self):
        dense=rows(range(7));sparse=rows([0,1,5,6])
        a=boundary.find_events(dense,min_frames=1)[0];b=boundary.find_events(sparse,min_frames=1,bridge=4)[0]
        boundary.score_event(a,dense,.1);boundary.score_event(b,sparse,.1)
        self.assertLess(b.confidence,a.confidence)
        self.assertAlmostEqual(b.evidence_coverage,4/7)
        self.assertEqual(a.confidence_terms['effective_frames'],1)

    def test_smoothing_never_crosses_gap(self):
        from vmax_vision.calib import Camera
        from vmax_vision.track_model import car_contacts
        cam=Camera.load_all()['trackside']
        detections={i:{'contacts_uv':cam.ground_to_pixel(car_contacts([-10+i*.1,6],0)), 'score':.9} for i in [0,1,2,30,31,32]}
        raw=boundary.judge_clip(detections,cam.H_inv,smooth=False)
        filtered=boundary.judge_clip(detections,cam.H_inv,smooth=True)
        np.testing.assert_allclose([f.position for f in raw],[f.position for f in filtered])

    def test_tyre_centre_outside_can_still_touch_line(self):
        lo,hi=clearance(np.array([[-10,7.05]]*4),0)
        self.assertTrue((lo<0).all());self.assertTrue((hi<0).all())
        lo,hi=clearance(np.array([[-10,7.30]]*4),0)
        self.assertTrue((lo>0).all());self.assertTrue(np.all(lo<=hi))

    def test_uncertainty_retains_systematic_error(self):
        r=frame_evidence(.04,.02,.05)
        self.assertEqual(r['geometry_status'],'uncertain')
        self.assertLess(r['margin_interval_m'][0],0)
        with self.assertRaises(ValueError):frame_evidence(.1,-.1)

    def test_cache_changes_with_video_weights_settings(self):
        with tempfile.TemporaryDirectory() as d:
            v,w=Path(d)/'v',Path(d)/'w';v.write_bytes(b'a');w.write_bytes(b'w')
            a=cache_signature(v,w,{'threshold':.2})
            v.write_bytes(b'b');b=cache_signature(v,w,{'threshold':.2})
            self.assertNotEqual(a,b)
            self.assertNotEqual(b,cache_signature(v,w,{'threshold':.3}))
            w.write_bytes(b'x');self.assertNotEqual(b,cache_signature(v,w,{'threshold':.2}))

    def test_scene_assignment_reproducible(self):
        a=make_specs(100,42);b=make_specs(100,42)
        self.assertEqual(a,b);self.assertEqual(len({s['family_id'] for s in a}),100)
        self.assertEqual({s['split'] for s in a},{'train','validation','test'})
        self.assertNotEqual(a,make_specs(100,43))

    def test_footprint_truth_differs_from_point_truth(self):
        ss=make_specs(1,5)[0];rs,_=trajectory(ss)
        self.assertTrue(all(r['footprint_margin_m']<r['point_margin_m'] for r in rs))

    def test_custom_surveyed_polygon(self):
        from vmax_vision.circuit import CircuitMap
        circuit=CircuitMap([[-20,-7],[20,-7],[20,7],[-20,7]])
        np.testing.assert_allclose(circuit.signed_excess([[0,0],[0,7],[0,7.1]]),[-7,0,.1],atol=1e-12)
        lo,hi=clearance(np.array([[0,7.05]]*4),0,track_map=circuit)
        self.assertTrue((hi<0).all())
        with self.assertRaises(ValueError):CircuitMap([[0,0],[1,1],[0,1],[1,0]])

    def test_video_byte_ranges(self):
        import functools,threading,http.client
        from http.server import ThreadingHTTPServer
        from vmax_vision.serve import RangeHandler
        with tempfile.TemporaryDirectory() as d:
            Path(d,'video.mp4').write_bytes(b'0123456789')
            server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(RangeHandler,directory=d))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for value,status,body in [('bytes=2-5',206,b'2345'),('bytes=-3',206,b'789'),('bytes=99-',416,b'')]:
                    conn=http.client.HTTPConnection(*server.server_address)
                    conn.request('GET','/video.mp4',headers={'Range':value})
                    response=conn.getresponse();self.assertEqual(response.status,status);self.assertEqual(response.read(),body);conn.close()
            finally:server.shutdown();server.server_close();thread.join()

    def test_path_escape_rejected(self):
        from vmax_vision.blind import resolve
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):resolve(d,'../labels.json')

    def test_blind_receipt_and_one_to_one_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v=root/'clip.mp4';v.write_bytes(b'test video identity; decoder not needed for scorer')
            c={'id':'abc','family_id':'family','split':'test','video':'clip.mp4','video_sha256':sha256(v),'fps':24,'frames':4,'calibration':{'homography':np.eye(3).tolist(),'sigma_m':0}}
            m=root/'manifest.json';m.write_text(json.dumps({'clips':[c]}))
            truth={'clips':{'abc':{'video_sha256':sha256(v),'frames':[{'frame_idx':i,'car_id':'car_1','world_position':[0,0],'point_is_violation':True,'point_margin_m':.2} for i in range(4)]}}}
            labels=root/'labels.json';labels.write_text(json.dumps(truth))
            event={'start_frame':0,'end_frame_inclusive':3}
            r={'clip_id':'abc','video_sha256':sha256(v),'tracks':{'1':{'frames':[{'frame_idx':i,'position_m':[0,0],'margin_m':.2} for i in range(4)],'events':[event,event]}}}
            seal_run(root/'run',m,{'abc':r},{'split':'test','geometry':'point'})
            report=score(m,root/'run',labels,root/'report')
            self.assertEqual(report['true_positives'],1);self.assertEqual(report['false_reports'],1)
            (root/'run'/'abc.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'prediction changed'):score(m,root/'run',labels,root/'report2')
            m.write_text(json.dumps({'clips':[c,{**c,'id':'xyz','split':'train'}]}))
            with self.assertRaisesRegex(ValueError,'crosses splits'):validate_manifest(m)

if __name__=='__main__':unittest.main()
