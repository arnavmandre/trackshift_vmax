"""Real (not mocked) end-to-end check: run the bridge script as a subprocess
against one of our own clips, using its own already-exported fast-model
predictions as the crop source, exactly as the server will invoke it."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLIND_ROOT = Path('C:/Users/arnav/trackshift_runs/kerb960/experiment/fresh_blind')
VMAX_LIVE = Path('C:/Users/arnav/trackshift_runs/kerb960/vmax_live')
CLIP_ID = '034764aba12aec43_a00'
CAMERA_JSON = VMAX_LIVE / f'{CLIP_ID}.camera.json'
FAST_PREDICTIONS = VMAX_LIVE / f'{CLIP_ID}.json'
CHECKPOINT = ROOT / 'training_runs' / 'boundary_heatmaps_20260912_191710_183913' / 'best_model'


class TestVmaxBridge(unittest.TestCase):
    def test_bridge_produces_a_verdict(self):
        video = BLIND_ROOT / CLIP_ID / 'original.mp4'
        for path in (video, CAMERA_JSON, FAST_PREDICTIONS, CHECKPOINT):
            if not path.exists():
                self.skipTest(f'Missing required local artifact: {path}')
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'vmax_bridge.py'),
             '--video', str(video), '--camera-json', str(CAMERA_JSON),
             '--fast-predictions', str(FAST_PREDICTIONS), '--checkpoint', str(CHECKPOINT)],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertIn('prediction', result)
        self.assertIn(result['prediction'], (True, False, None))
        self.assertEqual(result['frames_total'], 24)
        self.assertGreaterEqual(result['cars_seen'], 1)
        self.assertIsInstance(result['per_detection'], list)

    def test_bridge_reports_error_on_bad_video(self):
        for path in (CAMERA_JSON, FAST_PREDICTIONS, CHECKPOINT):
            if not path.exists():
                self.skipTest(f'Missing required local artifact: {path}')
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'vmax_bridge.py'),
             '--video', str(ROOT / 'does_not_exist.mp4'), '--camera-json', str(CAMERA_JSON),
             '--fast-predictions', str(FAST_PREDICTIONS), '--checkpoint', str(CHECKPOINT)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)
        result = json.loads(proc.stdout)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
