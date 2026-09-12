"""ByteTrack over VMAX-Net detections.

The point of tracking here is not pretty boxes: the sustained-frame rule that
separates a real excursion from a single-frame blip only means anything if the
frames being counted are known to belong to the *same car*. ByteTrack's second
association pass -- matching leftover tracks against the low-confidence
detections everyone else throws away -- is what keeps an identity alive through
the few frames where a car is small, partly occluded or dropping in score, and
so is what stops one excursion being reported as two.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou_matrix(a, b):
    """Pairwise IoU between two sets of xyxy boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    lt = np.maximum(a[..., :2], b[..., :2])
    rb = np.minimum(a[..., 2:], b[..., 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[..., 2:] - a[..., :2], 0, None), axis=-1)
    area_b = np.prod(np.clip(b[..., 2:] - b[..., :2], 0, None), axis=-1)
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def greedy_match(cost, threshold):
    """Greedy assignment. With at most a handful of cars, Hungarian buys nothing."""
    matches, used_r, used_c = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        if cost[r, c] > threshold:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(int(r))
        used_c.add(int(c))
        matches.append((int(r), int(c)))
    unmatched_r = [i for i in range(cost.shape[0]) if i not in used_r]
    unmatched_c = [j for j in range(cost.shape[1]) if j not in used_c]
    return matches, unmatched_r, unmatched_c


class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, w, h)."""

    def __init__(self, box, dt=1.0, process=1.6, measure=1.4):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], float)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 400.0, 400.0, 100.0, 100.0])
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt
        self.H = np.hstack([np.eye(4), np.zeros((4, 4))])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, process, process, process / 4, process / 4])
        self.R = np.diag([measure, measure, measure * 2, measure * 2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        z = np.array([cx, cy, box[2] - box[0], box[3] - box[1]], float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self):
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def velocity(self):
        return self.x[4:6].copy()


@dataclass
class Track:
    track_id: int
    kalman: KalmanBox
    score: float
    detection: dict
    state: str = "tentative"
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    history: list = field(default_factory=list)

    @property
    def box(self):
        return self.kalman.box


class ByteTrack:
    """Two-stage association: confident detections first, then the leftovers."""

    def __init__(self, high_thresh=0.45, low_thresh=0.12, match_thresh=0.75,
                 second_thresh=0.55, max_age=12, min_hits=2):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_thresh = second_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections, frame_idx=0):
        for track in self.tracks:
            track.kalman.predict()
            track.age += 1
            track.time_since_update += 1

        dets = [d for d in detections if d["score"] >= self.low_thresh]
        high = [d for d in dets if d["score"] >= self.high_thresh]
        low = [d for d in dets if d["score"] < self.high_thresh]

        pool = list(self.tracks)
        cost = 1.0 - iou_matrix([t.box for t in pool], [d["box"] for d in high])
        matches, unmatched_tracks, unmatched_dets = greedy_match(cost, 1.0 - (1.0 - self.match_thresh))
        for ti, di in matches:
            self._absorb(pool[ti], high[di], frame_idx)

        # Second pass: the low-scoring detections that a single-threshold
        # tracker discards are usually the same car, just briefly harder to see.
        remaining = [pool[i] for i in unmatched_tracks]
        cost2 = 1.0 - iou_matrix([t.box for t in remaining], [d["box"] for d in low])
        matches2, still_unmatched, _ = greedy_match(cost2, 1.0 - (1.0 - self.second_thresh))
        for ti, di in matches2:
            self._absorb(remaining[ti], low[di], frame_idx)

        for di in unmatched_dets:
            det = high[di]
            track = Track(self._next_id, KalmanBox(det["box"]), det["score"], det)
            track.history.append((frame_idx, det))
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        for t in self.tracks:
            if t.state == "tentative" and t.hits >= self.min_hits:
                t.state = "confirmed"
        return [t for t in self.tracks if t.state == "confirmed" and t.time_since_update == 0]

    def _absorb(self, track, det, frame_idx):
        track.kalman.update(det["box"])
        track.score = 0.6 * track.score + 0.4 * det["score"]
        track.detection = det
        track.hits += 1
        track.time_since_update = 0
        track.history.append((frame_idx, det))


def run(per_frame_detections, **kwargs):
    """Track a whole clip. Returns per-frame lists of ``(track_id, detection)``."""
    tracker = ByteTrack(**kwargs)
    out = []
    for idx, dets in enumerate(per_frame_detections):
        live = tracker.update(dets, idx)
        out.append([(t.track_id, t.detection, t.kalman.velocity) for t in live])
    return out, tracker
