"""CLI bridge: run the slow (Mask2Former boundary-heatmap) model on one clip
video, using OUR fast model's already-detected car locations to derive each
frame's crop instead of their own background-subtraction cropper (which
requires a camera that never moves between clips -- not true for this
simulator; see the spec's Amendment section for why).

Supports more than one detected car per frame: every observation in the fast
model's prediction JSON gets its own independent crop and its own
independent slow-model verdict, keyed by frame index and a within-frame
detection index (not by car identity, which the fast model does not track).

Always prints exactly one JSON object to stdout and exits 0 on success, 1 on
failure -- a caller only ever needs to read stdout, never stderr.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch

import boundary_training as bt

DEFAULT_CHECKPOINT = Path(__file__).parent / 'training_runs' / 'boundary_heatmaps_20260912_191710_183913' / 'best_model'
CROP_PADDING = 1.65  # matches boundary_training.Config.crop_padding's default


def crop_from_points(points_by_key, resolution):
    """One padded square (x, y, side) covering all given [x, y] points, same
    shape convention as boundary_training.find_car_crop's return value."""
    xs = [p[0] for p in points_by_key.values() if p is not None]
    ys = [p[1] for p in points_by_key.values() if p is not None]
    if not xs:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    side = max(64, int(round(max(w, h) * CROP_PADDING + 16)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width, height = resolution
    side = min(side, width, height)
    x = int(round(min(max(cx - side / 2, 0), width - side)))
    y = int(round(min(max(cy - side / 2, 0), height - side)))
    return (x, y, side)


def group_detections_by_frame(fast_predictions, fps):
    """Frame index -> [(car_id, points)] for the detections worth a deep look.

    Car identity comes from the fast model's tracker (exported as car_id). The
    order of detections within a frame is confidence order and changes between
    frames, so it is never used as identity. When the export has car identity,
    only tracked cars are examined: untracked detections are short fragments or
    duplicate boxes, and cropping them just adds noise. Older exports without
    car identity fall back to examining every detection.
    """
    observations = fast_predictions['observations']
    labelled = any(o.get('car_id') for o in observations)
    by_frame = defaultdict(list)
    for obs in observations:
        if labelled and not obs.get('car_id'):
            continue
        frame_idx = int(round(obs['time'] * fps))
        by_frame[frame_idx].append((obs.get('car_id'), obs['points']))
    return by_frame


def verdict(predictions):
    """Off track if any decided frame says so; on track if some decided and
    none did; None (inconclusive) if nothing was decided."""
    decided = [p for p in predictions if p is not None]
    if any(p is True for p in decided):
        return True
    return False if decided else None


@torch.no_grad()
def predict_one_crop(model, processor, image, crop, camera, config, device):
    cropped = bt.crop_image(image, crop, config.image_size)
    inputs = processor(images=cropped, return_tensors='pt')['pixel_values'].to(device)
    points, scores = bt.decode_heatmaps(model(inputs), config)
    points = bt.from_crop(points[0].cpu().numpy(), crop, config.image_size)
    scores = scores[0].cpu().numpy()
    prediction, interval, reason = bt.decide(points, scores, camera, config)
    return {'prediction': prediction, 'reason': reason}


def write_progress(progress_path, stage, done, total):
    """Best-effort progress for a caller polling alongside us. Never fatal --
    a progress write failing must not lose an otherwise-good inference run."""
    if not progress_path:
        return
    try:
        Path(progress_path).write_text(json.dumps({'stage': stage, 'done': done, 'total': total}))
    except OSError:
        pass


def run(video_path, camera, fast_predictions, checkpoint, device, progress_path=None):
    # Checkpoint load is a meaningful share of total runtime, so it gets its
    # own reported stage rather than sitting at 0% frame progress.
    write_progress(progress_path, 'loading_model', 0, 0)
    model, processor, config, _ = bt.load_checkpoint(Path(checkpoint), device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise RuntimeError('Could not determine video fps')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    by_frame = group_detections_by_frame(fast_predictions, fps)
    per_detection = []
    idx = 0
    try:
        write_progress(progress_path, 'processing', 0, total_frames)
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            detections = by_frame.get(idx, [])
            if detections:
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for car_id, points_by_key in detections:
                    crop = crop_from_points(points_by_key, camera['resolution'])
                    if crop is None:
                        per_detection.append({'frame': idx, 'car_id': car_id, 'prediction': None, 'reason': 'no_crop'})
                        continue
                    result = predict_one_crop(model, processor, image, crop, camera, config, device)
                    per_detection.append({'frame': idx, 'car_id': car_id, **result})
            idx += 1
            write_progress(progress_path, 'processing', idx, total_frames or idx)
    finally:
        cap.release()
    if not per_detection:
        raise RuntimeError('No fast-model detections to crop against in this clip')
    reasons = Counter(d['reason'] for d in per_detection if d.get('reason'))
    cars = sorted({d['car_id'] for d in per_detection if d['car_id']})
    # A separate verdict per car, so one car's excursion is never attributed to
    # another. The clip verdict is off track if any car is.
    per_car = {car: verdict([d['prediction'] for d in per_detection if d['car_id'] == car]) for car in cars}
    return {
        'prediction': verdict([d['prediction'] for d in per_detection]),
        'per_car': per_car,
        'cars_seen': len(cars),
        'frames_total': idx,
        'detections_total': len(per_detection),
        'detections_decided': sum(1 for d in per_detection if d['prediction'] is not None),
        'reasons': dict(reasons),
        'per_detection': per_detection,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--camera-json', required=True)
    p.add_argument('--fast-predictions', required=True)
    p.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--progress-file', default=None,
                   help='Optional path to write {"stage","done","total"} progress to while running')
    args = p.parse_args()
    try:
        camera = json.loads(Path(args.camera_json).read_text())
        fast_predictions = json.loads(Path(args.fast_predictions).read_text())
        result = run(args.video, camera, fast_predictions, args.checkpoint, args.device, args.progress_file)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}))
        sys.exit(1)
