"""Driver attribution ladder.

Knowing that *a* car left the track is worthless to a steward; the question is
always which one. The ladder is deliberately ordered from cheapest and most
reliable to weakest, and reports the highest rung that actually fired along with
what it was based on, so a marginal attribution is visible as marginal rather
than being laundered into a name.

  1. livery      colour signature inside the detected silhouette
  2. continuity  an unbroken track back to a frame where a higher rung fired
  3. lane        which side of the racing line the car has held
  4. sequence    running order along the track, when nothing else separates them

Rung 1 is the only one that identifies a car outright; the rest disambiguate
between candidates already on track, which is what the two-car scenarios need.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import track_model as tm

# Livery references, as rendered. A real deployment would load these from the
# entry list; here they come from the two liveries the simulator paints.
LIVERIES = {
    "car_1": {"name": "Car 1 (red)", "rgb": (218, 27, 39)},
    "car_2": {"name": "Car 2 (teal)", "rgb": (11, 141, 155)},
}

RUNGS = ["livery", "continuity", "lane", "sequence", "unattributed"]


@dataclass
class Attribution:
    track_id: int
    car_id: str | None
    rung: str
    confidence: float
    evidence: dict


def _hue_signature(frame_bgr, det, max_pixels=4000):
    """The car's own colour, taken from the most chromatic part of its silhouette.

    The circuit is not neutral -- red kerbs are within a few units of the red
    livery, and the painted apron is close to the teal one -- so a mean over the
    detection box reports the paint of whatever the car is standing on. Only
    masked pixels count, and of those only the most chromatic half, whose median
    is the paint rather than the shadow.
    """
    box = np.array(det["box"], float)
    x0, y0 = np.maximum(box[:2].astype(int), 0)
    x1 = min(int(box[2]) + 1, frame_bgr.shape[1])
    y1 = min(int(box[3]) + 1, frame_bgr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_bgr[y0:y1, x0:x1]
    sel = np.ones(patch.shape[:2], bool)
    crop = det.get("mask_crop")
    if crop is not None and np.size(crop["data"]):
        ox, oy = crop["origin"]
        data = np.asarray(crop["data"])
        sub = data[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        if sub.shape == sel.shape:
            sel = sub
    if crop is None or not np.size(crop.get("data", [])):
        return None                       # without a silhouette, do not guess
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sel = sel & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 45)
    if sel.sum() < 12:
        return None
    pixels = patch[sel].astype(np.float32)
    if len(pixels) > max_pixels:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), max_pixels, False)]
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab)
    chroma = np.linalg.norm(lab.reshape(-1, 3)[:, 1:].astype(np.float32) - 128.0, axis=1)
    keep = chroma >= np.median(chroma)
    return np.median(pixels[keep], axis=0)[::-1]  # BGR -> RGB


def _livery_match(signature):
    if signature is None:
        return None, 0.0, {}
    lab = {k: cv2.cvtColor(np.uint8([[v["rgb"][::-1]]]), cv2.COLOR_BGR2Lab)[0, 0].astype(float)
           for k, v in LIVERIES.items()}
    sig = cv2.cvtColor(np.uint8([[np.clip(signature[::-1], 0, 255)]]), cv2.COLOR_BGR2Lab)[0, 0].astype(float)
    # Chroma only: shading changes lightness far more than it changes hue.
    dist = {k: float(np.linalg.norm((sig - v)[1:])) for k, v in lab.items()}
    best, second = sorted(dist.items(), key=lambda kv: kv[1])[:2]
    separation = (second[1] - best[1]) / max(second[1], 1e-6)
    confidence = float(np.clip(separation * 1.6, 0.0, 1.0) * np.clip(1.0 - best[1] / 90.0, 0.0, 1.0))
    if best[1] > 70.0:
        return None, 0.0, {"distances": dist, "reason": "no livery within tolerance"}
    return best[0], confidence, {"distances": dist, "signature_rgb": [float(v) for v in signature]}


def _disjoint(a, b, tolerance=1):
    """True when two tracks never held the circuit at the same time."""
    if not a or not b:
        return True
    lo_a, hi_a = min(a), max(a)
    lo_b, hi_b = min(b), max(b)
    overlap = min(hi_a, hi_b) - max(lo_a, lo_b) + 1
    return overlap <= tolerance


def attribute(track_frames, frames_bgr, judgements_by_track, fps=24.0):
    """Attribute every track in a clip, walking down the ladder as needed.

    ``track_frames`` maps track id -> {frame index: detection}.
    """
    votes = {}
    for tid, per_frame in track_frames.items():
        picks, weights, evidence = [], [], []
        for idx, det in sorted(per_frame.items()):
            frame = frames_bgr.get(idx)
            if frame is None:
                continue
            sig = _hue_signature(frame, det)
            car, conf, ev = _livery_match(sig)
            if car and conf > 0.12:
                picks.append(car)
                weights.append(conf * det["score"])
                evidence.append(ev)
        votes[tid] = (picks, weights, evidence)

    attributions = {}
    claimed = {}
    order = sorted(track_frames, key=lambda t: -sum(votes[t][1]))
    for tid in order:
        picks, weights, _ev = votes[tid]
        if not picks:
            attributions[tid] = Attribution(tid, None, "unattributed", 0.0,
                                            {"reason": "no usable livery pixels"})
            continue
        tally = {}
        for car, w in zip(picks, weights):
            tally[car] = tally.get(car, 0.0) + w
        best = max(tally, key=tally.get)
        total = sum(tally.values())
        share = tally[best] / max(total, 1e-9)
        frames_voting = len(picks)
        confidence = float(np.clip(share * (1 - np.exp(-frames_voting / 12.0)), 0, 1))
        if best in claimed:
            # Two tracks claiming one car is only a conflict if both were on
            # track at the same time. Tracks that never coexist are one car the
            # tracker lost and picked up again -- that is the continuity rung,
            # and treating it as a conflict reports a single excursion twice.
            other = claimed[best]
            if _disjoint(track_frames[tid], track_frames[other]) and _motion_continuity(
                    judgements_by_track.get(tid, []), judgements_by_track.get(other, []), fps):
                attributions[tid] = Attribution(
                    tid, best, "continuity", float(np.clip(confidence, 0, 1)),
                    {"reason": f"resumes track {other}; the two never overlap",
                     "share": float(share), "voting_frames": frames_voting})
                continue
            attributions[tid] = Attribution(tid, None, "livery", 0.0,
                                            {"reason": f"{best} already claimed by track {other},"
                                                       " and both were on track together",
                                             "livery_vote": best, "share": float(share)})
            continue
        claimed[best] = tid
        attributions[tid] = Attribution(
            tid, best, "livery", confidence,
            {"share": float(share), "voting_frames": frames_voting,
             "tally": {k: float(v) for k, v in tally.items()}})

    # Lane and running order do not establish identity. Preserve unknowns.
    return attributions


def _motion_continuity(a, b, fps, max_gap_s=0.25, tolerance_m=1.0):
    """Conservative short-gap association using timestamped constant velocity."""
    if not a or not b or fps <= 0:
        return False
    a, b = sorted(a, key=lambda j: j.frame_idx), sorted(b, key=lambda j: j.frame_idx)
    if a[0].frame_idx > b[0].frame_idx:
        a, b = b, a
    dt = (b[0].frame_idx - a[-1].frame_idx) / fps
    if dt <= 0 or dt > max_gap_s or len(a) < 2:
        return False
    velocity = (a[-1].position - a[-2].position) / ((a[-1].frame_idx-a[-2].frame_idx)/fps)
    predicted = a[-1].position + velocity * dt
    return bool(np.linalg.norm(predicted-b[0].position) <= tolerance_m)
