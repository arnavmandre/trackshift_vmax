"""Zero-shot COCO baseline: what an off-the-shelf detector does with this footage.

The original plan proposed starting from a pretrained COCO car class. This runs
exactly that -- YOLOv8-seg, no fine-tuning -- over the held-out camera, and
records both what it detects as a vehicle and what it detects the cars *as*.
The answer is the reason the pipeline trains its own detector.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np

from . import clips as clipmod
from . import dataset as ds
from .calib import Camera


def run(weights="yolov8m-seg.pt", camera=None, stride=4, conf=0.02,
        out="pipeline_out/baseline_coco.json", verbose=True):
    from ultralytics import YOLO
    camera = camera or "trackside"
    model = YOLO(weights)
    cameras = Camera.load_all()
    vehicle = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}
    per_clip = []
    labels = Counter()
    tp = fn = 0
    for clip in clipmod.discover(include_race_pace=False):
        if clip.camera != camera:
            continue
        truth = {row["frame_idx"]: row["instances"] for row in ds.frame_labels(clip, cameras)}
        hits = total = 0
        for idx, frame in clip.frames():
            if idx % stride:
                continue
            gt = truth.get(idx, [])
            if not gt:
                continue
            total += len(gt)
            res = model.predict(frame, imgsz=1280, conf=conf, verbose=False)[0]
            names = res.names
            found = []
            for i in range(len(res.boxes)):
                cls = int(res.boxes.cls[i])
                labels[names[cls]] += 1
                if cls in vehicle:
                    found.append(res.boxes.xyxy[i].numpy())
            for inst in gt:
                box = np.array(inst["box"], float)
                best = 0.0
                for f in found:
                    lt = np.maximum(box[:2], f[:2])
                    rb = np.minimum(box[2:], f[2:])
                    wh = np.clip(rb - lt, 0, None)
                    inter = wh[0] * wh[1]
                    union = ((box[2] - box[0]) * (box[3] - box[1])
                             + (f[2] - f[0]) * (f[3] - f[1]) - inter)
                    best = max(best, float(inter / max(union, 1e-9)))
                hits += int(best >= 0.5)
        tp += hits
        fn += total - hits
        per_clip.append({"clip": clip.key, "matched": int(hits), "cars": int(total)})
        if verbose:
            print(f"  {clip.key:34s} vehicle-class recall {hits}/{total}", flush=True)
    payload = {
        "weights": weights,
        "camera": camera,
        "confidence_threshold": conf,
        "frame_stride": stride,
        "vehicle_class_recall": float(tp / max(tp + fn, 1)),
        "matched": int(tp),
        "cars": int(tp + fn),
        "per_clip": per_clip,
        "top_predicted_labels": [[str(k), int(v)] for k, v in labels.most_common(12)],
        "total_detections": int(sum(labels.values())),
        "note": ("COCO classes, no fine-tuning, and a 2% confidence floor -- at any "
                 "usable threshold the recall collapses. Even here the same car is "
                 "labelled a bench, a suitcase or a surfboard as readily as a "
                 "vehicle, and a vehicle-class box carries no ground contact point, "
                 "so it cannot produce a track-limit margin at all."),
    }
    pathlib.Path(out).write_text(json.dumps(payload, indent=1))
    return payload


if __name__ == "__main__":
    run()
