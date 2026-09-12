# Accurate-Model Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the slower, more accurate Mask2Former/Swin-Tiny boundary model into the steward UI, with automatic escalation for low-confidence fast-model clips, a manual per-clip trigger button, and an agreement-based confidence badge.

**Architecture:** A subprocess bridge script runs the slow model (in its own isolated venv) on one clip at a time; `vmax_live_server.py` gains a background FIFO worker that auto-enqueues low-confidence clips and serves two new endpoints for the UI's manual button/popup to poll.

**Tech Stack:** Python stdlib (`http.server`, `threading`, `queue`, `subprocess`) on the server side; the slow model's own stack (`torch`, `transformers`, `opencv-python-headless`) runs isolated inside `vmax_model2/Track_limit_detection/.venv_bench`; plain JS in the existing steward-UI prototype.

**Spec:** `docs/superpowers/specs/2026-09-13-accurate-model-cascade-design.md`

## Global Constraints

- Auto-escalation threshold: mean fast-model confidence **< 0.75** (per clip, same metric the UI already shows).
- Slow-model jobs run **sequentially**, one at a time (a single background worker thread + `queue.Queue`), never in parallel.
- The slow model's own venv (`vmax_model2/Track_limit_detection/.venv_bench`) must never be merged into or installed alongside the main repo's venv — always invoked via `subprocess`.
- Camera angle in scope for this pass: **`a00` only**. Do not build backgrounds for `a01`-`a03`.
- Every new script must be runnable standalone with `--help` and sensible defaults matching the paths already used elsewhere in this session (see each task).
- No placeholder/fake verdicts on failure — a bridge/subprocess error must surface as an explicit `error` status, never a fabricated result.

---

## File Structure

- Modify `vmax_export.py` — write a `<clip_id>.camera.json` sidecar (the clip's `camera_spec`, already homography-compatible) alongside the existing prediction JSON/video, so the server never needs to reach outside its own data directory.
- Create `vmax_model2/Track_limit_detection/vmax_background.py` — one-off script, builds a median background plate for our `a00` camera angle from existing clip videos.
- Create `vmax_model2/Track_limit_detection/vmax_bridge.py` — CLI: video + camera JSON + background + checkpoint in, one JSON verdict out (stdout), exit code signals success/failure.
- Create `vmax_model2/Track_limit_detection/tests/test_vmax_background.py` and `tests/test_vmax_bridge.py` — real (not mocked) checks against the actual checkpoint and a real clip.
- Modify `vmax_live_server.py` — background worker/queue, startup auto-escalation scan, `POST`/`GET /api/better/<id>`, disk-cached results.
- Create `test_vmax_live_server.py` (repo root, matching the existing flat test-file convention) — starts the real server as a subprocess, hits the new endpoints.
- Modify `VMAXPROTO/VMAX-Steward-Review-v2.1.html` — button, popup `<dialog>`, polling, agreement badge.
- Verification: a Playwright script (written to the scratchpad at execution time, not committed — matches how every other UI change this session was verified).

---

## Task 1: `vmax_export.py` — camera_spec sidecar

**Files:**
- Modify: `vmax_export.py` (the `export_clip` function and the `__main__` loop)

**Interfaces:**
- Produces: `<out_dir>/<clip_id>.camera.json` — the exact `c['camera_spec']` dict from the blind manifest, unmodified. Consumed by Task 3's server code and Task 2's bridge script.

- [ ] **Step 1: Add the sidecar write to `export_clip`**

Open `vmax_export.py`. In `export_clip(model, device, clip_entry, threshold, out_dir, display_name)`, right after the existing lines that write `(out_dir / f"{clip_entry['id']}.json")` and copy the video, add:

```python
    (out_dir / f"{clip_entry['id']}.camera.json").write_text(json.dumps(clip_entry['camera_spec'], indent=2))
```

- [ ] **Step 2: Re-run the export for the current 32-clip set**

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift"
.venv/Scripts/python.exe vmax_export.py
```

Expected: same "VMAX EXPORT COMPLETE ... 32 clips" output as before, plus new `.camera.json` files.

- [ ] **Step 3: Verify**

```bash
ls "C:/Users/arnav/trackshift_runs/kerb960/vmax_live"/*.camera.json | wc -l
```

Expected: `32`.

```bash
.venv/Scripts/python.exe -c "import json; d=json.loads(open(r'C:/Users/arnav/trackshift_runs/kerb960/vmax_live/034764aba12aec43_a00.camera.json').read()); print(d['name'], d['resolution'], 'ground_plane_homography' in d)"
```

Expected: `034764aba12aec43_a00 [960, 540] True`.

- [ ] **Step 4: Commit**

```bash
git add vmax_export.py
git commit -m "Export each clip's camera_spec as a sidecar JSON for the slow-model bridge"
```

(Do not commit the regenerated `vmax_live/` output — it's outside the repo, under `trackshift_runs/`.)

---

## Task 2: Background plate for camera angle `a00`

**Files:**
- Create: `vmax_model2/Track_limit_detection/vmax_background.py`
- Create: `vmax_model2/Track_limit_detection/tests/test_vmax_background.py`

**Interfaces:**
- Produces: a background PNG file (RGB, same resolution as our clips: 960x540). Consumed by Task 3 (server passes its path to the bridge) and Task 2's own test.

- [ ] **Step 1: Write the script**

Create `vmax_model2/Track_limit_detection/vmax_background.py`:

```python
"""Builds one median background plate for our simulator's 'a00' camera angle,
for the slow model's background-subtraction car cropper (boundary_training.
find_car_crop). Run once; re-run only if the chosen camera angle changes.

Samples frames from our own existing blind-test clip videos -- the same
median-background method boundary_training.build_backgrounds() uses on its
own project's data, just pointed at ours.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_BLIND_ROOT = Path('C:/Users/arnav/trackshift_runs/kerb960/experiment/fresh_blind')
DEFAULT_OUTPUT = Path('C:/Users/arnav/trackshift_runs/kerb960/vmax_live/bettermodel_background_a00.png')


def build_background(video_paths, sample_count=12):
    frames = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_path}')
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ids = np.unique(np.linspace(0, count - 1, min(count, sample_count), dtype=int))
            for frame_id in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {video_path}:{frame_id}')
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    if not frames:
        raise ValueError('No frames collected')
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--blind-root', type=Path, default=DEFAULT_BLIND_ROOT)
    p.add_argument('--angle-suffix', default='_a00')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    video_paths = sorted(args.blind_root.glob(f'*{args.angle_suffix}/original.mp4'))
    if not video_paths:
        raise SystemExit(f'No videos matched *{args.angle_suffix}/original.mp4 under {args.blind_root}')
    background = build_background(video_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(background).save(args.output)
    print(f'Wrote background from {len(video_paths)} videos -> {args.output}')
```

- [ ] **Step 2: Write the test**

Create `vmax_model2/Track_limit_detection/tests/test_vmax_background.py`:

```python
"""Real (not mocked) check: build a background from two of our own clips and
confirm it's a plausible image, not that it matches any exact prior output."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmax_background import build_background, DEFAULT_BLIND_ROOT


class TestVmaxBackground(unittest.TestCase):
    def test_build_background_shape_and_range(self):
        videos = sorted(DEFAULT_BLIND_ROOT.glob('*_a00/original.mp4'))[:2]
        if len(videos) < 2:
            self.skipTest('Fewer than 2 a00 clips available locally')
        background = build_background(videos, sample_count=4)
        self.assertEqual(background.dtype, np.uint8)
        self.assertEqual(background.shape, (540, 960, 3))
        self.assertTrue((background > 0).any())


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it fails (module doesn't exist yet — skip if already created)**

Since Step 1 already created the module, instead run it directly to confirm behavior:

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift/vmax_model2/Track_limit_detection"
.venv_bench/Scripts/python.exe -m unittest tests.test_vmax_background -v
```

Expected: `test_build_background_shape_and_range ... ok` (or a skip message if clips aren't present).

- [ ] **Step 4: Actually build the real background plate**

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift/vmax_model2/Track_limit_detection"
.venv_bench/Scripts/python.exe vmax_background.py
```

Expected: `Wrote background from 40 videos -> C:/Users/arnav/trackshift_runs/kerb960/vmax_live/bettermodel_background_a00.png` (40 = one per blind incident with an `a00` angle).

- [ ] **Step 5: Visually sanity-check it**

Read the PNG (e.g. via the Read tool) and confirm it looks like an empty track corner with no sharp car silhouette (faint ghosting from the median is expected and fine).

- [ ] **Step 6: Commit**

```bash
git add vmax_model2/Track_limit_detection/vmax_background.py vmax_model2/Track_limit_detection/tests/test_vmax_background.py
git commit -m "Add background-plate builder for the slow model's car cropper"
```

(The generated PNG itself lives under `trackshift_runs/`, outside the repo — not committed.)

---

## Task 3: `vmax_bridge.py` — the subprocess bridge

**Files:**
- Create: `vmax_model2/Track_limit_detection/vmax_bridge.py`
- Create: `vmax_model2/Track_limit_detection/tests/test_vmax_bridge.py`

**Interfaces:**
- Consumes: `boundary_training.load_checkpoint(path, device) -> (model, processor, config, backgrounds)`; `boundary_training.predict_frame(model, processor, image, camera, backgrounds, config, device) -> {'prediction': bool|None, 'reason': str|None, ...}`; `boundary_training.read_rgb(path) -> np.ndarray`.
- Produces: one JSON object on stdout — `{"prediction": true|false|null, "frames_total": int, "frames_decided": int, "reasons": {...}, "per_frame": [{"frame": int, "prediction": ..., "reason": ...}, ...]}` on success (exit 0), or `{"error": "..."}` (exit 1) on failure. This exact shape is what Task 5's server code parses.

- [ ] **Step 1: Write the script**

Create `vmax_model2/Track_limit_detection/vmax_bridge.py`:

```python
"""CLI bridge: run the slow (Mask2Former boundary-heatmap) model on one clip
video, using this project's own camera_spec/background-plate format. Always
prints exactly one JSON object to stdout and exits 0 on success, 1 on
failure -- a caller only ever needs to read stdout, never stderr, to get a
result or a clear error.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch

import boundary_training as bt

DEFAULT_CHECKPOINT = Path(__file__).parent / 'training_runs' / 'boundary_heatmaps_20260912_191710_183913' / 'best_model'


def run(video_path, camera, background, checkpoint, device):
    model, processor, config, _ = bt.load_checkpoint(Path(checkpoint), device)
    backgrounds = {camera['name']: background}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {video_path}')
    per_frame = []
    try:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = bt.predict_frame(model, processor, image, camera, backgrounds, config, device)
            per_frame.append({'frame': idx, 'prediction': result.get('prediction'), 'reason': result.get('reason')})
            idx += 1
    finally:
        cap.release()
    if not per_frame:
        raise RuntimeError('Video decoded zero frames')
    decided = [f for f in per_frame if f['prediction'] is not None]
    if any(f['prediction'] is True for f in decided):
        prediction = True
    elif decided:
        prediction = False
    else:
        prediction = None
    reasons = Counter(f['reason'] for f in per_frame if f['reason'])
    return {
        'prediction': prediction,
        'frames_total': len(per_frame),
        'frames_decided': len(decided),
        'reasons': dict(reasons),
        'per_frame': per_frame,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--camera-json', required=True)
    p.add_argument('--background', required=True)
    p.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    try:
        camera = json.loads(Path(args.camera_json).read_text())
        background = bt.read_rgb(Path(args.background))
        result = run(args.video, camera, background, args.checkpoint, args.device)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}))
        sys.exit(1)
```

- [ ] **Step 2: Write the test**

Create `vmax_model2/Track_limit_detection/tests/test_vmax_bridge.py`:

```python
"""Real (not mocked) end-to-end check: run the bridge script as a subprocess
against one of our own clips and the background plate from Task 2, exactly
as the server will invoke it. Requires the checkpoint and background plate
to already exist locally -- skips cleanly if they don't."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLIND_ROOT = Path('C:/Users/arnav/trackshift_runs/kerb960/experiment/fresh_blind')
CAMERA_JSON = Path('C:/Users/arnav/trackshift_runs/kerb960/vmax_live/034764aba12aec43_a00.camera.json')
BACKGROUND = Path('C:/Users/arnav/trackshift_runs/kerb960/vmax_live/bettermodel_background_a00.png')
CHECKPOINT = ROOT / 'training_runs' / 'boundary_heatmaps_20260912_191710_183913' / 'best_model'


class TestVmaxBridge(unittest.TestCase):
    def test_bridge_produces_a_verdict(self):
        video = BLIND_ROOT / '034764aba12aec43_a00' / 'original.mp4'
        for path in (video, CAMERA_JSON, BACKGROUND, CHECKPOINT):
            if not path.exists():
                self.skipTest(f'Missing required local artifact: {path}')
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'vmax_bridge.py'),
             '--video', str(video), '--camera-json', str(CAMERA_JSON),
             '--background', str(BACKGROUND), '--checkpoint', str(CHECKPOINT)],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertIn('prediction', result)
        self.assertIn(result['prediction'], (True, False, None))
        self.assertEqual(result['frames_total'], 24)
        self.assertIsInstance(result['per_frame'], list)
        self.assertEqual(len(result['per_frame']), 24)

    def test_bridge_reports_error_on_bad_video(self):
        for path in (CAMERA_JSON, BACKGROUND, CHECKPOINT):
            if not path.exists():
                self.skipTest(f'Missing required local artifact: {path}')
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'vmax_bridge.py'),
             '--video', str(ROOT / 'does_not_exist.mp4'), '--camera-json', str(CAMERA_JSON),
             '--background', str(BACKGROUND), '--checkpoint', str(CHECKPOINT)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)
        result = json.loads(proc.stdout)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 3: Run the test — must be run with `.venv_bench`'s own interpreter (`sys.executable` inside the test resolves correctly only when the test itself runs under that interpreter)**

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift/vmax_model2/Track_limit_detection"
.venv_bench/Scripts/python.exe -m unittest tests.test_vmax_bridge -v
```

Expected: both tests `ok` (or skipped if Task 2's background plate / the checkpoint aren't present — resolve that before continuing if so).

- [ ] **Step 4: Time it, for the cascade design's sequencing assumption**

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift/vmax_model2/Track_limit_detection"
.venv_bench/Scripts/python.exe -c "
import time, subprocess, sys
t0 = time.perf_counter()
subprocess.run([sys.executable, 'vmax_bridge.py',
    '--video', r'C:/Users/arnav/trackshift_runs/kerb960/experiment/fresh_blind/034764aba12aec43_a00/original.mp4',
    '--camera-json', r'C:/Users/arnav/trackshift_runs/kerb960/vmax_live/034764aba12aec43_a00.camera.json',
    '--background', r'C:/Users/arnav/trackshift_runs/kerb960/vmax_live/bettermodel_background_a00.png'],
    check=True)
print('seconds:', time.perf_counter() - t0)
"
```

Record the printed time in the task's completion notes — this confirms (or corrects) the spec's "~28x slower than the fast model" estimate for a single-camera clip and matters for how the server's background queue is explained to the user later.

- [ ] **Step 5: Commit**

```bash
git add vmax_model2/Track_limit_detection/vmax_bridge.py vmax_model2/Track_limit_detection/tests/test_vmax_bridge.py
git commit -m "Add subprocess bridge to the slow model, single-view predict_frame per clip"
```

---

## Task 4: `vmax_live_server.py` — background worker + endpoints

**Files:**
- Modify: `vmax_live_server.py`

**Interfaces:**
- Consumes: Task 1's `<clip_id>.camera.json`, Task 2's background PNG, Task 3's `vmax_bridge.py` (invoked via `.venv_bench/Scripts/python.exe`).
- Produces: `POST /api/better/<clip_id>` → `{"status": "queued"|"already queued"|"already running"|"already done"}`; `GET /api/better/<clip_id>` → `{"status": "queued"|"running"|"done"|"error"|"not_started", "result": <bridge JSON>|null}`. Both consumed by Task 5's UI code. Also writes `<data_dir>/<clip_id>.bettermodel.json` as the on-disk cache (`{"status": "done"|"error", "result": {...}}`).

- [ ] **Step 1: Add the new imports and constants**

At the top of `vmax_live_server.py`, after the existing imports, add:

```python
import queue
import re
import subprocess
import sys
import threading

MODEL2_ROOT = REPO / 'vmax_model2' / 'Track_limit_detection'
BETTER_PYTHON = MODEL2_ROOT / '.venv_bench' / 'Scripts' / 'python.exe'
BETTER_BRIDGE = MODEL2_ROOT / 'vmax_bridge.py'
BETTER_BACKGROUND = None  # set from --data at startup; see __main__
CONFIDENCE_ESCALATION_THRESHOLD = 0.75
CLIP_ID_RE = re.compile(r'^[A-Za-z0-9_]+$')

better_jobs = {}          # clip_id -> "queued" | "running" | "done" | "error"
better_jobs_lock = threading.Lock()
better_queue = queue.Queue()
```

- [ ] **Step 2: Write the job-processing helpers (module-level functions)**

Add these functions after the constants, before the `Handler` class:

```python
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
    background = data_dir / 'bettermodel_background_a00.png'
    result_path = data_dir / f'{clip_id}.bettermodel.json'
    if not camera_json.is_file() or not video.is_file() or not background.is_file():
        payload = {'status': 'error', 'result': {'error': 'missing camera_spec, video, or background plate'}}
        result_path.write_text(json.dumps(payload))
        return payload
    proc = subprocess.run(
        [str(BETTER_PYTHON), str(BETTER_BRIDGE),
         '--video', str(video), '--camera-json', str(camera_json), '--background', str(background)],
        capture_output=True, text=True, cwd=str(MODEL2_ROOT), timeout=180)
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
```

- [ ] **Step 3: Add `do_POST` and the two new `do_GET` routes to `Handler`**

In `Handler`, add a `do_POST` method (alongside the existing `do_GET`):

```python
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
```

In `Handler.do_GET`, add a new branch (before the final `else: self.send_error(404)`):

```python
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
                self.respond_bytes(json.dumps(payload).encode(), 'application/json')
```

- [ ] **Step 4: Wire up the worker thread and startup auto-escalation in `__main__`**

Replace the body of `if __name__ == '__main__':` with:

```python
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
    escalated = 0
    for c in manifest['clips']:
        score = mean_confidence(data_dir, c['id'])
        if score is not None and score < CONFIDENCE_ESCALATION_THRESHOLD:
            enqueue_better(data_dir, c['id'])
            escalated += 1
    if escalated:
        print(f'Auto-escalated {escalated} low-confidence clip(s) to the accurate model.', flush=True)

    server = ThreadingHTTPServer(('127.0.0.1', a.port), functools.partial(Handler, data_dir=data_dir))
    print(f"Open http://127.0.0.1:{a.port} — friend's prototype UI with live model predictions auto-loaded. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
```

- [ ] **Step 5: Write the server test**

Create `test_vmax_live_server.py` at the repo root:

```python
"""Starts the real vmax_live_server.py as a subprocess and exercises the new
/api/better endpoints against real data -- no mocking of the HTTP layer or
the job queue."""
import json
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

DATA_DIR = Path('C:/Users/arnav/trackshift_runs/kerb960/vmax_live')
PORT = 8011  # different port so it never collides with a manually-running dev server


class TestVmaxLiveServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA_DIR / 'clips_manifest.json').is_file():
            raise unittest.SkipTest('vmax_export.py has not been run locally')
        cls.proc = subprocess.Popen(
            [sys.executable, 'vmax_live_server.py', '--data', str(DATA_DIR), '--port', str(PORT)],
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

    def test_enqueue_and_poll_a_clip(self):
        manifest = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/clips').read())
        clip_id = manifest['clips'][0]['id']

        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/api/better/{clip_id}', method='POST')
        first = json.loads(urllib.request.urlopen(req).read())
        self.assertIn(first['status'], ('queued', 'already queued', 'already running', 'already done'))

        status = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/better/{clip_id}').read())
        self.assertIn(status['status'], ('queued', 'running', 'done', 'error', 'not_started'))

    def test_unknown_clip_id_is_rejected(self):
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/api/better/not a valid id', method='POST')
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 6: Run the test**

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift"
.venv/Scripts/python.exe -m unittest test_vmax_live_server -v
```

Expected: both tests `ok`.

- [ ] **Step 7: Manual smoke check against the real running server**

If a dev server is already running on port 8010 from earlier in this session, stop it first (find and kill the process), then:

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift"
.venv/Scripts/python.exe vmax_live_server.py
```

Expected startup line: either no "Auto-escalated" line (if all 32 clips are ≥0.75 mean confidence) or `Auto-escalated N low-confidence clip(s) to the accurate model.` — check the printed number is plausible given the confidence scores already shown in the UI queue (all 32 clips were previously seen scoring 76-97%, so 0 auto-escalations is the likely, correct outcome for this exact set — this is expected, not a bug, unless the model was re-exported differently since).

- [ ] **Step 8: Commit**

```bash
git add vmax_live_server.py test_vmax_live_server.py
git commit -m "Add background worker, auto-escalation, and /api/better endpoints to the live server"
```

---

## Task 5: UI — button, popup, agreement badge

**Files:**
- Modify: `VMAXPROTO/VMAX-Steward-Review-v2.1.html`

**Interfaces:**
- Consumes: `POST /api/better/<id>`, `GET /api/better/<id>` from Task 4.
- Consumes existing: `current()`, `analysed(c)`, `$(id)`, `choose(id)`, `list()` (all already defined in this file per this session's earlier work).
- Produces: nothing new consumed elsewhere — this is the UI leaf.

- [ ] **Step 1: Add the button and popup dialog markup**

In `VMAXPROTO/VMAX-Steward-Review-v2.1.html`, find the `.clip-heading` div (the one containing `id="clipTitle"` and the "Remove clip" button) and add a new button next to "Remove clip":

```html
<button class="link" id="runBetterModel">Run through accurate model</button>
```

Immediately before the closing `</body>` marker the server injects `autoload.js` at (i.e. right after the existing `<dialog id="removeDialog">...</dialog>` block), add a new dialog:

```html
<dialog id="betterModelDialog">
 <h2>Accurate model result</h2>
 <div id="betterModelLoading">
  <p>Running <strong id="betterModelClipName"></strong> through the accurate model. This is much slower than the fast model — typically several seconds per clip.</p>
  <div style="width:100%;height:8px;background:#1c1c1d;border:1px solid #2e2e30;border-radius:4px;overflow:hidden;margin-top:14px">
   <div id="betterModelBar" style="height:100%;width:0%;background:#d90d17;transition:width .3s linear"></div>
  </div>
 </div>
 <div id="betterModelResult" hidden>
  <p id="betterModelVerdict"></p>
  <p id="betterModelDetail" class="muted small"></p>
 </div>
 <div id="betterModelError" hidden class="wb-error"></div>
 <div class="dialog-actions"><button data-close="betterModelDialog">Close</button></div>
</dialog>
```

- [ ] **Step 2: Add the polling/trigger JS**

Find the existing wiring section (near `$('removeClip').onclick=...` and `$('confirmRemove').onclick=...`) and add:

```js
function betterVerdictText(prediction) {
 if (prediction === true) return 'OFF TRACK';
 if (prediction === false) return 'IN TRACK';
 return 'INCONCLUSIVE';
}
async function pollBetterModel(clipId) {
 for (let i = 0; i < 200; i++) { // up to ~200 * 1.5s = 5 minutes
  const status = await (await fetch(`/api/better/${clipId}`)).json();
  if (status.status === 'done' || status.status === 'error') return status;
  const pct = Math.min(92, 8 + i * 2); // indeterminate-ish creep, never claims false completion
  $('betterModelBar').style.width = pct + '%';
  await new Promise(res => setTimeout(res, 1500));
 }
 return { status: 'error', result: { error: 'Timed out waiting for a result' } };
}
async function runBetterModelForCurrentClip() {
 const c = current();
 if (!c) return;
 $('betterModelClipName').textContent = c.name;
 $('betterModelLoading').hidden = false;
 $('betterModelResult').hidden = true;
 $('betterModelError').hidden = true;
 $('betterModelBar').style.width = '4%';
 $('betterModelDialog').showModal();
 await fetch(`/api/better/${c.id}`, { method: 'POST' });
 const status = await pollBetterModel(c.id);
 $('betterModelLoading').hidden = true;
 if (status.status === 'error') {
  $('betterModelError').hidden = false;
  $('betterModelError').textContent = 'Could not get a result from the accurate model: ' +
   (status.result?.error || 'unknown error');
  return;
 }
 c.betterModel = status.result;
 $('betterModelResult').hidden = false;
 $('betterModelVerdict').textContent = 'Accurate model: ' + betterVerdictText(status.result.prediction);
 $('betterModelDetail').textContent =
  `${status.result.frames_decided} / ${status.result.frames_total} frames reached a decision.`;
 renderBetterModelBadge(c);
 list();
}
$('runBetterModel').onclick = runBetterModelForCurrentClip;
```

- [ ] **Step 3: Add the agreement badge**

Add a new function, and call it from `choose()` (so it renders whenever a clip becomes active) and from the end of `runBetterModelForCurrentClip` (already added above):

```js
function renderBetterModelBadge(c) {
 let el = document.getElementById('betterModelBadge');
 if (!el) {
  el = document.createElement('p');
  el.id = 'betterModelBadge';
  el.style.cssText = 'font-size:11px;font-weight:600;margin-top:4px';
  $('playbackStatus').insertAdjacentElement('afterend', el);
 }
 if (!c?.betterModel || !analysed(c)) { el.textContent = ''; return; }
 const fastOff = c.evidence.original.candidates.length > 0;
 const slowPrediction = c.betterModel.prediction;
 if (slowPrediction === null) {
  el.textContent = 'Accurate model inconclusive — low confidence';
  el.style.color = '#ffbd69';
 } else if (fastOff === slowPrediction) {
  el.textContent = 'Models agree — high confidence';
  el.style.color = '#7df5c1';
 } else {
  el.textContent = 'Models disagree — low confidence';
  el.style.color = '#ffbd69';
 }
}
```

In `choose(id)`, immediately after the existing `updatePlaybackStatus(c);` line, add:

```js
 renderBetterModelBadge(c);
```

- [ ] **Step 4: Commit**

```bash
git add VMAXPROTO/VMAX-Steward-Review-v2.1.html
git commit -m "Add manual accurate-model trigger button, popup, and agreement badge to the UI"
```

---

## Task 6: End-to-end verification

**Files:**
- None (verification only — a scratchpad Playwright script, not committed, matching how every other UI change in this session was checked)

**Interfaces:**
- Consumes: the running server from Task 4/5 at `http://127.0.0.1:8010`.

- [ ] **Step 1: Start the server fresh**

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift"
.venv/Scripts/python.exe vmax_live_server.py
```

- [ ] **Step 2: Write and run a Playwright check**

Write to the scratchpad (e.g. `<scratchpad>/check_better_model.js`):

```js
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('pageerror', err => errors.push('pageerror: ' + err.message));

  await page.goto('http://127.0.0.1:8010', { waitUntil: 'load', timeout: 20000 });
  await page.waitForFunction(() => !document.getElementById('vmaxLoadingGate'), null, { timeout: 60000 });

  await page.click('#runBetterModel');
  await page.waitForSelector('#betterModelDialog[open]', { timeout: 5000 });
  const loadingVisible = await page.evaluate(() => !document.getElementById('betterModelLoading').hidden);

  await page.waitForFunction(() => {
    const d = document.getElementById('betterModelDialog');
    return !document.getElementById('betterModelLoading').hidden === false;
  }, null, { timeout: 180000 });

  const resultHidden = await page.evaluate(() => document.getElementById('betterModelResult').hidden);
  const errorHidden = await page.evaluate(() => document.getElementById('betterModelError').hidden);
  const verdictText = await page.textContent('#betterModelVerdict').catch(() => null);
  const badgeText = await page.evaluate(() => document.getElementById('betterModelBadge')?.textContent);

  console.log('loadingVisible (should be true right after click):', loadingVisible);
  console.log('resultHidden:', resultHidden, '| errorHidden:', errorHidden);
  console.log('verdictText:', verdictText);
  console.log('badgeText (agreement/disagreement):', badgeText);
  console.log('consoleErrors:', JSON.stringify(errors));

  await page.screenshot({ path: 'screenshot_better_model.png' });
  await browser.close();
})();
```

Run it with `node check_better_model.js` from the scratchpad directory (the same `playwright` package already installed there earlier this session).

- [ ] **Step 3: Confirm the result**

Expected: `loadingVisible: true`, exactly one of `resultHidden`/`errorHidden` is `false` (never both hidden, never both shown), `verdictText` contains `OFF TRACK`/`IN TRACK`/`INCONCLUSIVE`, `badgeText` contains `agree` or `disagree` or `inconclusive`, `consoleErrors: []`. Look at the screenshot to confirm it visually matches this session's established UI patterns (dark theme, consistent styling with the other dialogs).

- [ ] **Step 4: Confirm auto-escalation didn't break normal load**

Reload the page once more and confirm the existing loading gate / 32-clip queue / live overlay behavior (all built earlier this session) still works exactly as before — the new startup auto-escalation scan must not delay or interfere with the existing clip-loading gate.

---

## Self-review notes (for whoever executes this)

- **Spec coverage**: background plate (Task 2), bridge script (Task 3), server worker/auto-escalation/endpoints (Task 4), UI button/popup/badge (Task 5), testing (Task 6) — every section of the spec has a task.
- **Type consistency**: `run_better_model`'s return shape (`{"status", "result"}`) matches what `do_GET`'s `/api/better/` branch reads, matches what the UI's `pollBetterModel`/`runBetterModelForCurrentClip` expect (`status.status`, `status.result.prediction`, `status.result.frames_decided`, `status.result.frames_total`, `status.result.error`) — checked consistent across Tasks 3-5.
- **Known risk to watch during execution**: Task 3 Step 4's timing measurement may reveal the single-view bridge is faster or slower than the spec's ~28x estimate (that estimate was extrapolated from their 3-camera benchmark, not measured directly for our single-camera case). If it's dramatically different, that's fine — it doesn't change the design (still sequential, still a background queue) — just update the number if it's mentioned to the user later.

## Amendment (during execution, after Task 2): background plate dropped, Task 3 redesigned

Task 2 was completed and immediately found unusable: our simulator randomizes the camera per incident even at the same angle index, so a background built by averaging many incidents' `a00` frames is a blurry composite of different framings, not one static scene — their `find_car_crop()` requires a camera that never moves between clips. Full detail and reasoning: see the spec's own "Amendment" section (`docs/superpowers/specs/2026-09-13-accurate-model-cascade-design.md`).

**Task 2 is dropped entirely** — `vmax_background.py`/its test/the generated PNG were deleted. No background plate is needed.

**Task 3 is redesigned**: the bridge derives each frame's crop from our own fast model's already-detected tyre keypoints (bounding box of all 4 points, padded 1.65x — matching `boundary_training.Config.crop_padding`'s default) instead of their background-subtraction cropper, calling the slow model's public lower-level functions directly (`crop_image`, `decode_heatmaps`, `from_crop`, `decide`) instead of `predict_frame()`. This also adds multi-car support: any frame with more than one fast-model detection gets one independent crop + verdict per detection, keyed by `(frame, car_index)`.

**Interface change**: bridge CLI arg `--background <path>` becomes `--fast-predictions <path>` (our own exported `<clip_id>.json`). Output shape: `per_frame`/`frames_decided` become `per_detection`/`detections_decided`/`detections_total`, plus a new `cars_seen` field. `result.prediction` (`true`/`false`/`null`) is unchanged in meaning and location — Task 4's server-side JSON handling needs no change beyond the CLI arg it passes; Task 5's UI needs `frames_decided`/`frames_total` reads changed to `detections_decided`/`detections_total`.

The actual implementation code for the redesigned Task 3 (script + test) was written directly during execution rather than re-drafted here first — see `vmax_model2/Track_limit_detection/vmax_bridge.py` and its test for the final, as-run version.
