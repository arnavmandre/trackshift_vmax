"""Adapter: export our trained model's predictions as vmax.predictions.v1 JSON,
consumed by the VMAXPROTO steward-review prototype's window.VMAX.importPredictions().

Reuses experiment.py's own observation/margin logic (same math the blind
evaluation trusts, using each clip's exact camera calibration) so the emitted
violation candidates mean the same thing they meant in the evaluation report.
Produces no model output of its own; this is purely a format adapter.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path('C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift')))
import experiment as e

BLIND_ROOT = Path('C:/Users/arnav/trackshift_runs/kerb960/experiment/fresh_blind')
SELECTION = Path('C:/Users/arnav/trackshift_runs/kerb960/experiment/selection.json')
WEIGHTS = Path('C:/Users/arnav/trackshift_runs/kerb960/experiment/selected.pt')
OUT_DIR = Path('C:/Users/arnav/trackshift_runs/kerb960/vmax_live')

# The prototype's UI names tyres FL/FR/RL/RR; our model's keypoint channels are
# ordered per PLAN['keypoints']. Map by name, not position, so a future keypoint
# reorder can't silently swap tyres in the exported evidence.
KEYPOINT_TO_TYRE = {'front_left': 'FL', 'front_right': 'FR', 'rear_left': 'RL', 'rear_right': 'RR'}
TYRE_ORDER = [KEYPOINT_TO_TYRE[k] for k in e.PLAN['keypoints']]


def infer_raw(model, video_path, device):
    rows = []
    for i, r in enumerate(model.predict(source=str(video_path), stream=True, device=device,
                    imgsz=e.PLAN['imgsz'], conf=.1, max_det=8, verbose=False)):
        points = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.empty((0, 4, 3))
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        rows.append({'frame': i, 'detections': [{'score': float(s), 'box': b.tolist(), 'keypoints': p.tolist()}
                for s, b, p in zip(scores, boxes, points)]})
    return rows


def export_clip(model, device, clip_entry, threshold, out_dir, display_name):
    video_path = BLIND_ROOT / clip_entry['video']
    raw = infer_raw(model, video_path, device)
    fps = clip_entry['fps']

    observations = []
    for frame in raw:
        for j, d in enumerate(frame['detections']):
            kp = d['keypoints']
            points = {tyre: [round(kp[idx][0], 2), round(kp[idx][1], 2)] for idx, tyre in enumerate(TYRE_ORDER)}
            observations.append({
                'id': f"observation-{frame['frame']}-{j}",
                'time': round(frame['frame'] / fps, 4),
                'car_id': None,
                'confidence': round(d['score'], 4),
                'points': points,
            })

    # Same margin/track logic the blind evaluation itself uses, at the same
    # selected threshold, so "candidate" here means what it meant in that report.
    tracks = e.observations(clip_entry, raw, threshold)
    candidates = []
    for t_idx, track in enumerate(tracks):
        violated_frames = (f['frame'] for f in track if f['margin_m'] > 0)
        for g_idx, (start_f, end_f) in enumerate(e.groups(violated_frames)):
            candidates.append({
                'id': f'event-{t_idx}-{g_idx}',
                'start': round(start_f / fps, 4),
                'end': round(end_f / fps, 4),
            })

    # "fast model": our YOLO26n-pose CNN, to distinguish it later from the
    # transformer model planned as a low-confidence fallback (see the
    # cascade design discussion) once both are wired into this same UI.
    payload = {
        'schema': 'vmax.predictions.v1',
        'source': 'model',
        'model_version': f"YOLO26n-pose (fast model) selected.pt@{e.digest(WEIGHTS)[:12]}",
        'video': {'name': f"{display_name}.mp4", 'width': e.PLAN['width'], 'height': e.PLAN['height'],
                   'duration': round(clip_entry['frames'] / fps, 4)},
        'observations': observations,
        'candidates': candidates,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip_entry['id']}.json").write_text(json.dumps(payload, indent=2))
    (out_dir / f"{clip_entry['id']}.camera.json").write_text(json.dumps(clip_entry['camera_spec'], indent=2))
    shutil.copy2(video_path, out_dir / f"{clip_entry['id']}.mp4")
    print(f"{display_name} ({clip_entry['id']}): {len(observations)} observations, {len(candidates)} candidates", flush=True)


if __name__ == '__main__':
    manifest = json.loads((BLIND_ROOT / 'manifest.json').read_text())
    selection = json.loads(SELECTION.read_text())
    if e.digest(WEIGHTS) != selection['weights_sha256']:
        raise ValueError('selected model changed since selection.json was written')
    threshold = selection['winner']['metrics']['threshold']

    # Kept small deliberately while the model/UI interaction is still being
    # worked out -- eager-loads all of these behind one loading screen. Raise
    # this once that's solid; see README note in VMAXPROTO/ for the tradeoff.
    MAX_INCIDENTS = 8  # 32 clips

    clips = manifest['clips']
    # Stable, human-readable numbering: incidents ordered by family_id, angles
    # within an incident ordered by angle_index -- independent of manifest order.
    family_ids = sorted({c['family_id'] for c in clips})
    incident_number = {fam: i + 1 for i, fam in enumerate(family_ids)}
    clips_sorted = sorted(
        (c for c in clips if incident_number[c['family_id']] <= MAX_INCIDENTS),
        key=lambda c: (incident_number[c['family_id']], c['angle_index']))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(str(WEIGHTS))
    print(f'device={device} threshold={threshold} weights={WEIGHTS} clips={len(clips_sorted)}', flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_meta = []
    for c in clips_sorted:
        display_name = f"Incident {incident_number[c['family_id']]} Clip {c['angle_index'] + 1}"
        export_clip(model, device, c, threshold, OUT_DIR, display_name)
        clips_meta.append({'id': c['id'], 'name': f'{display_name}.mp4', 'fps': c['fps']})
    (OUT_DIR / 'clips_manifest.json').write_text(json.dumps({'clips': clips_meta}, indent=2))
    print('VMAX EXPORT COMPLETE', OUT_DIR, len(clips_meta), 'clips', flush=True)
