"""Steward review footage: the pipeline's own findings burned onto the clip.

Every mark drawn here comes from the pipeline, not from the labels -- the
projected track limit uses the homography the pipeline judged with, the contact
dots are the ones the network predicted, and the margin and confidence readouts
are the values the report will carry. If the overlay looks wrong, the verdict
is wrong, which is the point of producing it.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np

from . import track_model as tm
from .calib import apply_h

FONT = cv2.FONT_HERSHEY_DUPLEX
INK = (238, 240, 244)
PANEL = (24, 28, 34)
GOOD = (150, 235, 120)
WARN = (60, 205, 255)
BAD = (72, 84, 255)
LIMIT = (60, 210, 250)


def _polyline(frame, H, world_xy, colour, thickness=2):
    """Draw a ground curve, breaking it where it passes behind the camera.

    The homography's last row is the projective depth: without this test the
    track limit reappears mirrored above the horizon, which looks like a
    detection error and is not one.
    """
    world_xy = np.asarray(world_xy, float)
    depth = world_xy @ np.asarray(H, float)[2, :2] + H[2, 2]
    pts = apply_h(H, world_xy)
    ok = (depth > 1e-6) & np.isfinite(pts).all(axis=1) & (np.abs(pts) < 1e5).all(axis=1)
    run = []
    for p, good in zip(pts, ok):
        if good:
            run.append(p)
        elif len(run) >= 2:
            cv2.polylines(frame, [np.array(run, np.int32)], False, colour, thickness, cv2.LINE_AA)
            run = []
        else:
            run = []
    if len(run) >= 2:
        cv2.polylines(frame, [np.array(run, np.int32)], False, colour, thickness, cv2.LINE_AA)


def _panel(frame, x, y, w, h, alpha=0.72):
    sub = frame[y:y + h, x:x + w]
    if sub.size:
        frame[y:y + h, x:x + w] = (sub * (1 - alpha) + np.array(PANEL) * alpha).astype(np.uint8)


def _bar(frame, x, y, w, h, value, colour):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 76, 84), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * np.clip(value, 0, 1)), y + h), colour, -1)


def render_clip(clip, result, out_path, homography, show_limit=True, quality=20):
    """Write an annotated MP4 for one clip from one pipeline result."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = np.asarray(homography, float)

    tracks = result["tracks"]
    by_frame = {}
    for tid, track in tracks.items():
        for f in track["frames"]:
            by_frame.setdefault(f["frame_idx"], []).append((tid, track, f))
    events = {tid: track["events"] for tid, track in tracks.items()}

    limit_left = tm.boundary_polyline(1, n=900)
    limit_right = tm.boundary_polyline(-1, n=900)

    writer = None
    for idx, frame in clip.frames():
        frame = frame.copy()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     clip.fps, (w, h))
        if show_limit:
            _polyline(frame, H, limit_left, LIMIT, 2)
            _polyline(frame, H, limit_right, LIMIT, 2)

        rows = by_frame.get(idx, [])
        for tid, track, f in rows:
            colour = BAD if f["is_violation"] else (GOOD if f["margin_m"] < -0.25 else WARN)
            contacts = apply_h(H, np.array(f["contacts_world"]))
            for (u, v), e in zip(contacts, f["excess_m"]):
                cv2.circle(frame, (int(u), int(v)), 5, BAD if e > 0 else GOOD, -1, cv2.LINE_AA)
            hull = cv2.convexHull(contacts.astype(np.float32)).astype(np.int32)
            cv2.polylines(frame, [hull], True, colour, 2, cv2.LINE_AA)
            label = (track.get("attribution") or {}).get("car_id") or f"track {tid}"
            anchor = (int(contacts[:, 0].min()), int(contacts[:, 1].min()) - 10)
            cv2.putText(frame, f"{label}  {f['margin_m']:+.3f} m", anchor,
                        FONT, 0.52, colour, 1, cv2.LINE_AA)

        _panel(frame, 18, 18, 596, 30 + 34 * max(len(rows), 1) + 26)
        cv2.putText(frame, "VMAX STEWARD  |  track-limit assessment", (32, 44),
                    FONT, 0.56, INK, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{result['scenario']}  {result['camera']}  "
                           f"[{result['calibration_label']}]  frame {idx}",
                    (32, 66), FONT, 0.42, (168, 176, 188), 1, cv2.LINE_AA)
        y = 92
        for tid, track, f in rows:
            live = [e for e in events.get(tid, [])
                    if e["start_frame"] <= idx <= e["end_frame_inclusive"]]
            label = (track.get("attribution") or {}).get("car_id") or f"track {tid}"
            verdict = "POSSIBLE EXCURSION" if f["is_violation"] else "within limits"
            colour = BAD if f["is_violation"] else GOOD
            cv2.putText(frame, f"{label:<8s} margin {f['margin_m']:+.3f} m   {verdict}",
                        (32, y), FONT, 0.46, colour, 1, cv2.LINE_AA)
            if live:
                conf = live[0]["confidence"]
                _bar(frame, 32, y + 8, 260, 6, conf, BAD if conf > 0.5 else WARN)
                cv2.putText(frame, f"review candidate, score {conf*100:.0f}%",
                            (302, y + 14), FONT, 0.40, INK, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "no offence reported for this frame", (32, y + 14),
                            FONT, 0.40, (150, 158, 170), 1, cv2.LINE_AA)
            y += 34
        if not rows:
            cv2.putText(frame, "no car tracked in this frame", (32, y),
                        FONT, 0.44, (150, 158, 170), 1, cv2.LINE_AA)
        writer.write(frame)
    if writer is not None:
        writer.release()
    return out_path
