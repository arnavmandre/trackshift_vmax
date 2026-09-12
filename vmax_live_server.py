"""Serves the VMAXPROTO steward-review prototype with real model predictions
auto-loaded, via the prototype's own documented integration surface.

The prototype's HTML file on disk is never modified: it is read fresh and one
<script src="/autoload.js"> tag is appended before </body> at serve time.
Standard library only, matching review_server.py's approach.
"""
import argparse
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import re
import subprocess
import threading
from urllib.parse import unquote, urlsplit

REPO = Path('C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift')
PROTO_HTML = REPO / 'VMAXPROTO' / 'VMAX-Steward-Review-v2.1.html'
AUTOLOAD_JS = REPO / 'VMAXPROTO' / 'autoload.js'

MODEL2_ROOT = REPO / 'vmax_model2' / 'Track_limit_detection'
BETTER_PYTHON = MODEL2_ROOT / '.venv_bench' / 'Scripts' / 'python.exe'
BETTER_BRIDGE = MODEL2_ROOT / 'vmax_bridge.py'
CONFIDENCE_ESCALATION_THRESHOLD = 0.80
CLIP_ID_RE = re.compile(r'^[A-Za-z0-9_]+$')

# Shared across every Handler instance (ThreadingHTTPServer makes a new one
# per request) and the single background worker thread.
better_jobs = {}          # clip_id -> "queued" | "running" | "done" | "error"
better_jobs_lock = threading.Lock()
better_queue = queue.Queue()


def mean_confidence(data_dir, clip_id):
    prediction_path = data_dir / f'{clip_id}.json'
    if not prediction_path.is_file():
        return None
    doc = json.loads(prediction_path.read_text())
    scores = [o['confidence'] for o in doc.get('observations', []) if o.get('confidence') is not None]
    return sum(scores) / len(scores) if scores else None


def run_better_model(data_dir, clip_id):
    camera_json = data_dir / f'{clip_id}.camera.json'
    video = data_dir / f'{clip_id}.mp4'
    fast_predictions = data_dir / f'{clip_id}.json'
    result_path = data_dir / f'{clip_id}.bettermodel.json'
    progress_path = data_dir / f'{clip_id}.bettermodel.progress'
    if not camera_json.is_file() or not video.is_file() or not fast_predictions.is_file():
        payload = {'status': 'error', 'result': {'error': 'missing camera_spec, video, or fast-model predictions'}}
        result_path.write_text(json.dumps(payload))
        return payload
    try:
        proc = subprocess.run(
            [str(BETTER_PYTHON), str(BETTER_BRIDGE),
             '--video', str(video), '--camera-json', str(camera_json),
             '--fast-predictions', str(fast_predictions),
             '--progress-file', str(progress_path)],
            capture_output=True, text=True, cwd=str(MODEL2_ROOT), timeout=180)
    finally:
        progress_path.unlink(missing_ok=True)
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed = {'error': f'bridge produced no valid JSON (exit {proc.returncode}): {proc.stderr[-2000:]}'}
    status = 'error' if proc.returncode != 0 or 'error' in parsed else 'done'
    payload = {'status': status, 'result': parsed}
    result_path.write_text(json.dumps(payload))
    return payload


def better_worker_loop(data_dir):
    while True:
        clip_id = better_queue.get()
        try:
            with better_jobs_lock:
                better_jobs[clip_id] = 'running'
            payload = run_better_model(data_dir, clip_id)
            with better_jobs_lock:
                better_jobs[clip_id] = payload['status']
        except Exception as exc:
            with better_jobs_lock:
                better_jobs[clip_id] = 'error'
            (data_dir / f'{clip_id}.bettermodel.json').write_text(
                json.dumps({'status': 'error', 'result': {'error': str(exc)}}))
        finally:
            better_queue.task_done()


def enqueue_better(data_dir, clip_id):
    result_path = data_dir / f'{clip_id}.bettermodel.json'
    with better_jobs_lock:
        current = better_jobs.get(clip_id)
        if current in ('queued', 'running'):
            return f'already {current}'
        if result_path.is_file() and current != 'error':
            return 'already done'
        better_jobs[clip_id] = 'queued'
    better_queue.put(clip_id)
    return 'queued'


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, data_dir, **kwargs):
        self.data_dir = data_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/':
                html = PROTO_HTML.read_text(encoding='utf-8')
                html = html.replace('</body>', '<script src="/autoload.js"></script></body>')
                self.respond_bytes(html.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/autoload.js':
                self.respond_bytes(AUTOLOAD_JS.read_bytes(), 'application/javascript')
            elif path == '/api/clips':
                manifest = json.loads((self.data_dir / 'clips_manifest.json').read_text())
                for c in manifest['clips']:
                    c['video_url'] = f"/video/{c['id']}.mp4"
                    c['predictions_url'] = f"/predictions/{c['id']}.json"
                self.respond_bytes(json.dumps(manifest).encode(), 'application/json')
            elif path.startswith('/video/'):
                self.respond_file(self.data_dir / path[len('/video/'):], 'video/mp4')
            elif path.startswith('/predictions/'):
                self.respond_file(self.data_dir / path[len('/predictions/'):], 'application/json')
            elif path.startswith('/api/better/'):
                clip_id = path[len('/api/better/'):]
                if not CLIP_ID_RE.match(clip_id):
                    self.send_error(400)
                    return
                result_path = self.data_dir / f'{clip_id}.bettermodel.json'
                if result_path.is_file():
                    payload = json.loads(result_path.read_text())
                else:
                    with better_jobs_lock:
                        state = better_jobs.get(clip_id, 'not_started')
                    payload = {'status': state, 'result': None}
                    if state == 'running':
                        # Real per-frame progress from the bridge, when it has
                        # got far enough to report any -- never a fake estimate.
                        try:
                            payload['progress'] = json.loads(
                                (self.data_dir / f'{clip_id}.bettermodel.progress').read_text())
                        except (OSError, json.JSONDecodeError):
                            pass
                self.respond_bytes(json.dumps(payload).encode(), 'application/json')
            else:
                self.send_error(404)
        except FileNotFoundError:
            self.send_error(404)

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        if path.startswith('/api/better/'):
            clip_id = path[len('/api/better/'):]
            if not CLIP_ID_RE.match(clip_id):
                self.send_error(400)
                return
            status = enqueue_better(self.data_dir, clip_id)
            self.respond_bytes(json.dumps({'status': status}).encode(), 'application/json')
        else:
            self.send_error(404)

    def respond_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def respond_file(self, path, content_type):
        if not path.is_file() or not path.resolve().is_relative_to(self.data_dir.resolve()):
            self.send_error(404)
            return
        self.respond_bytes(path.read_bytes(), content_type)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=r'C:/Users/arnav/trackshift_runs/kerb960/vmax_live')
    p.add_argument('--port', type=int, default=8010)
    a = p.parse_args()
    data_dir = Path(a.data)
    if not (data_dir / 'clips_manifest.json').is_file():
        raise SystemExit(f'No clips_manifest.json in {data_dir} — run vmax_export.py first')

    threading.Thread(target=better_worker_loop, args=(data_dir,), daemon=True).start()

    manifest = json.loads((data_dir / 'clips_manifest.json').read_text())
    low_confidence = queued = 0
    for c in manifest['clips']:
        score = mean_confidence(data_dir, c['id'])
        if score is not None and score < CONFIDENCE_ESCALATION_THRESHOLD:
            low_confidence += 1
            # Count what was actually queued, not what merely qualified --
            # already-cached results are skipped and must not be reported as work.
            if enqueue_better(data_dir, c['id']) == 'queued':
                queued += 1
    if low_confidence:
        print(f'{low_confidence} clip(s) below the {CONFIDENCE_ESCALATION_THRESHOLD:.0%} confidence score threshold; '
              f'{queued} queued for the deep model, {low_confidence - queued} already cached.', flush=True)

    server = ThreadingHTTPServer(('127.0.0.1', a.port), functools.partial(Handler, data_dir=data_dir))
    print(f"Open http://127.0.0.1:{a.port} — friend's prototype UI with live model predictions auto-loaded. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
