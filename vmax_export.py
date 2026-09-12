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


# Tracks shorter than this are brief fragments (a car entering/leaving view, a
# one-off duplicate box), not a separate car; they get no car label.
MIN_TRACK_OBSERVATIONS = 3


def track_progress(x, y):
    """Distance along the corner's centreline (metres) for a world position:
    entry straight (x < 0), 40 m-radius arc, then the exit straight (y > 40)."""
    if x < 0:
        return x
    if y > 40:
        return 20 * np.pi + (y - 40)
    return 40 * np.arctan2(x, 40 - y)


def build_evidence(raw, calibration_entry, threshold, fps):
    """Observations and candidate windows for one clip, with stable car identity.

    Car identity comes from the same tracker the blind evaluation uses
    (experiment.observations), which follows each car across frames. Detection
    order within a frame is by confidence and can flip between frames, so it
    must never be used as identity.

    Cars are named by position on track -- "Car 1" is the car furthest ahead --
    using world coordinates rather than image order. Every camera angle of an
    incident sees the same world, so the same physical car gets the same name in
    every clip of that incident.
    """
    # Same margin/track logic the blind evaluation itself uses, at the same
    # selected threshold, so "candidate" here means what it meant in that report.
    tracks = e.observations(calibration_entry, raw, threshold)
    cars = [t for t in tracks if len(t) >= MIN_TRACK_OBSERVATIONS]
    # Compare cars only at frames where they are visible together: each camera
    # sees a car for a different stretch of the clip, so averaging a car's
    # progress over its own visible frames would rank cars differently per
    # camera. A car's lead is its progress relative to the other cars present.
    by_frame = {}
    for i, t in enumerate(cars):
        for ob in t:
            by_frame.setdefault(ob['frame'], {})[i] = track_progress(*ob['position'])
    lead = [[] for _ in cars]
    for present in by_frame.values():
        if len(present) > 1:
            mean = np.mean(list(present.values()))
            for i, s in present.items():
                lead[i].append(s - mean)
    order = sorted(range(len(cars)), key=lambda i: (-np.mean(lead[i]) if lead[i] else 0.0, cars[i][0]['frame']))
    cars = [cars[i] for i in order]
    label = {id(t): f'Car {i + 1}' for i, t in enumerate(cars)}
    car_of_detection, clearance_of_detection = {}, {}
    for track in tracks:
        for ob in track:
            key = (ob['frame'], tuple(ob['box']))
            car_of_detection[key] = label.get(id(track))
            # Distance from the nearest track edge in metres, as measured by the
            # evaluation pipeline (positive = past the line). Only tracked
            # detections at or above the operating threshold get one.
            clearance_of_detection[key] = round(ob['margin_m'], 3)

    observations = []
    for frame in raw:
        for j, d in enumerate(frame['detections']):
            kp = d['keypoints']
            points = {tyre: [round(kp[idx][0], 2), round(kp[idx][1], 2)] for idx, tyre in enumerate(TYRE_ORDER)}
            observations.append({
                'id': f"observation-{frame['frame']}-{j}",
                'time': round(frame['frame'] / fps, 4),
                'car_id': car_of_detection.get((frame['frame'], tuple(d['box']))),
                'clearance_m': clearance_of_detection.get((frame['frame'], tuple(d['box']))),
                'confidence': round(d['score'], 4),
                'points': points,
            })

    candidates = []
    for t_idx, track in enumerate(tracks):
        violated_frames = (f['frame'] for f in track if f['margin_m'] > 0)
        for g_idx, (start_f, end_f) in enumerate(e.groups(violated_frames)):
            candidates.append({
                'id': f'event-{t_idx}-{g_idx}',
                'start': round(start_f / fps, 4),
                'end': round(end_f / fps, 4),
                'car_id': label.get(id(track)),
            })
    return observations, candidates


def raw_from_prediction(prediction, fps):
    """Rebuild tracker input from an already-exported prediction file, so car
    identity can be added to existing exports without re-running the model."""
    by_frame = {}
    for ob in prediction['observations']:
        keypoints = [[*ob['points'][tyre], 1.0] for tyre in TYRE_ORDER]
        xs, ys = [p[0] for p in keypoints], [p[1] for p in keypoints]
        by_frame.setdefault(int(round(ob['time'] * fps)), []).append(
            {'score': ob['confidence'], 'box': [min(xs), min(ys), max(xs), max(ys)], 'keypoints': keypoints})
    frames = range(int(round(prediction['video']['duration'] * fps)))
    return [{'frame': i, 'detections': by_frame.get(i, [])} for i in frames]


def relabel_export_dir(directory, threshold):
    """Add car identity to every prediction file in an existing export folder."""
    directory = Path(directory)
    manifest = json.loads((directory / 'clips_manifest.json').read_text())
    for clip in manifest['clips']:
        path = directory / f"{clip['id']}.json"
        prediction = json.loads(path.read_text())
        camera = json.loads((directory / f"{clip['id']}.camera.json").read_text())
        entry = {'calibration': {'homography': camera['ground_plane_homography']}}
        observations, candidates = build_evidence(raw_from_prediction(prediction, clip['fps']), entry, threshold, clip['fps'])
        before = len(prediction['candidates'])
        prediction['observations'], prediction['candidates'] = observations, candidates
        path.write_text(json.dumps(prediction, indent=2))
        cars = sorted({o['car_id'] for o in observations if o['car_id']})
        print(f"{clip['name']}: cars={cars} candidates {before}->{len(candidates)}", flush=True)


def export_clip(model, device, clip_entry, threshold, out_dir, display_name):
    video_path = BLIND_ROOT / clip_entry['video']
    raw = infer_raw(model, video_path, device)
    fps = clip_entry['fps']
    observations, candidates = build_evidence(raw, clip_entry, threshold, fps)

    # "fast model": our YOLO26n-pose CNN, to distinguish it later from the
    # transformer model planned as a low-confidence fallback (see the
    # cascade design discussion) once both are wired into this same UI.
    payload = {
        'schema': 'vmax.predictions.v1',
        'source': 'model',
        'model_version': f"YOLO26n-pose (fast model) selected.pt@{e.digest(WEIGHTS)[:12]}",
        'video': {'name': f"{display_name}.mp4", 'width': clip_entry['camera_spec']['resolution'][0], 'height': clip_entry['camera_spec']['resolution'][1],
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--relabel', metavar='DIR',
                        help='add car identity to an existing export folder instead of exporting')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='detection threshold for --relabel (the selected operating threshold)')
    cli = parser.parse_args()
    if cli.relabel:
        relabel_export_dir(cli.relabel, cli.threshold)
        sys.exit(0)

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
