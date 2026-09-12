"""Runtime integration checks for CI/installed machines; no accuracy claims."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    import cv2
    import torch
except ImportError:
    cv2 = torch = None


@unittest.skipIf(cv2 is None or torch is None,'requires OpenCV and PyTorch')
class RuntimeTests(unittest.TestCase):
    def test_manifest_training_loader_and_backward(self):
        from vmax_vision.dataset import build_manifest
        from vmax_vision.traindata import CropSampler
        from vmax_vision.train import to_torch
        from vmax_vision.model import VmaxNet, detection_loss
        samples=build_manifest('examples/generated_dataset/manifest.json',
                               'examples/generated_dataset/sealed/labels.json',max_frames=4,frame_stride=4)
        self.assertTrue(samples)
        manifest=json.loads(Path('examples/generated_dataset/manifest.json').read_text())
        allowed={c['id'] for c in manifest['clips'] if c['split']=='train'}
        self.assertTrue(all(s['clip'] in allowed for s in samples))
        torch.set_num_threads(2)
        x,target=to_torch(CropSampler(samples,crop=128,seed=21).batch(2))
        loss,_=detection_loss(VmaxNet()(x),target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_checkpoint_inference(self):
        from vmax_vision.detector import VmaxDetector
        from vmax_vision.model import VmaxNet
        with tempfile.TemporaryDirectory() as d:
            weights=Path(d)/'model.pt'
            torch.save({'model':VmaxNet().state_dict(),'step':0},weights)
            detector=VmaxDetector(weights=str(weights),threshold=.99,device='cpu')
            result=detector(np.zeros((128,128,3),dtype=np.uint8))
            self.assertIsInstance(result,list)

    def test_real_video_cache_invalidates_on_replacement(self):
        from vmax_vision import pipeline
        from vmax_vision.clips import Clip
        class FakeDetector:
            name='test';weights=None;threshold=.25;calls=0
            def __call__(self,frame):
                self.calls+=1
                return []
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'source.mp4';old=pipeline.DETECTION_CACHE;pipeline.DETECTION_CACHE=d/'cache'
            def write(value):
                w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),12,(64,64))
                for _ in range(3):w.write(np.full((64,64,3),value,np.uint8))
                w.release()
            try:
                write(0);c=Clip('test','camera',video,None,fps=12);det=FakeDetector()
                pipeline.detect_clip(c,det);self.assertEqual(det.calls,3)
                pipeline.detect_clip(c,det);self.assertEqual(det.calls,3)
                write(200);pipeline.detect_clip(c,det);self.assertEqual(det.calls,6)
                det.threshold=.3;pipeline.detect_clip(c,det);self.assertEqual(det.calls,9)
            finally:pipeline.DETECTION_CACHE=old

if __name__=='__main__':unittest.main()
