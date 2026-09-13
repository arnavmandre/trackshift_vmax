"""Starts the real vmax_live_server.py as a subprocess and exercises the new
/api/better endpoints against real data -- no mocking of the HTTP layer or
the job queue."""
import json
import shutil
import subprocess
import tempfile
import sys
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
PORT = 8011  # different port so it never collides with a manually-running dev server


class TestVmaxLiveServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (REPO / 'final_demo' / 'clips_manifest.json').is_file():
            raise unittest.SkipTest('final_demo/ not present')
        # Serve a copy so queued deep-model jobs never write into the bundled demo.
        cls.tmp = tempfile.TemporaryDirectory()
        data_dir = Path(cls.tmp.name) / 'data'
        shutil.copytree(REPO / 'final_demo', data_dir)
        cls.proc = subprocess.Popen(
            [sys.executable, 'vmax_live_server.py', '--data', str(data_dir), '--port', str(PORT)],
            cwd=str(Path(__file__).parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(50):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/clips', timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            cls.proc.terminate()
            raise RuntimeError('Server did not start in time')

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.proc.stdout.close()
        cls.tmp.cleanup()

    def test_enqueue_and_poll_a_clip(self):
        manifest = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/clips').read())
        clip_id = manifest['clips'][0]['id']

        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/api/better/{clip_id}', method='POST')
        first = json.loads(urllib.request.urlopen(req).read())
        self.assertIn(first['status'], ('queued', 'already queued', 'already running', 'already done'))

        status = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/better/{clip_id}').read())
        self.assertIn(status['status'], ('queued', 'running', 'done', 'error', 'not_started'))

    def test_unknown_clip_id_is_rejected(self):
        bad_id = urllib.parse.quote('not a valid id')
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/api/better/{bad_id}', method='POST')
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
