"""Clip discovery and frame decoding.

Everything downstream treats a clip as an ordinary video file plus the name of
the camera that shot it. Ground-truth JSON is loaded separately and only by the
training and evaluation code.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DATA = ROOT / "data"

FIXED_CAMERAS = ["trackside", "exit", "broadcast"]
SCENARIOS = ["clean_lap", "near_miss", "violation_0.05m", "violation_0.15m",
             "violation_0.30m", "violation_0.60m", "side_by_side", "sustained_vs_blip"]

GRADUATED = ["violation_0.05m", "violation_0.15m", "violation_0.30m", "violation_0.60m"]

# Leave one camera out *per scenario*, plus two scenarios held out entirely.
#
# Every clip that carries a headline number was therefore never trained on,
# while all three viewpoints still appear somewhere in training. Holding one
# camera out globally instead was tried and does not work here: the trackside
# rig puts cars up to 2150 px wide against a maximum of 466 px anywhere else,
# and two static viewpoints plus one moving rig are too thin to learn viewpoint
# invariance from -- that model reached 0.80 centre confidence on the cameras it
# had seen and 0.11 on the one it had not.
HELD_OUT_BY_SCENARIO = {
    "clean_lap": "broadcast",
    "near_miss": "exit",
    "violation_0.05m": "trackside",
    "violation_0.15m": "exit",
    "violation_0.30m": "broadcast",
    "violation_0.60m": "trackside",
    "side_by_side": "*",        # unseen traffic situation, from every angle
    "sustained_vs_blip": "*",
}
TRAIN_CAMERAS = FIXED_CAMERAS
HELD_OUT_SCENARIOS = [k for k, v in HELD_OUT_BY_SCENARIO.items() if v == "*"]


def is_held_out(scenario, camera):
    """True when this exact clip was withheld from training."""
    held = HELD_OUT_BY_SCENARIO.get(scenario)
    return held == "*" or held == camera


@dataclass
class Clip:
    scenario: str
    camera: str
    video: pathlib.Path
    labels: pathlib.Path
    fps: float = 24.0
    moving_camera: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.scenario}/{self.camera}"

    def is_training(self):
        if self.moving_camera:
            return True
        return not is_held_out(self.scenario, self.camera)

    @property
    def held_out(self):
        return not self.moving_camera and is_held_out(self.scenario, self.camera)

    def ground_truth(self):
        return json.loads(self.labels.read_text())

    def frames(self, limit=None):
        yield from read_frames(self.video, limit)


def discover(output=OUTPUT, data=DATA, include_race_pace=True):
    """All renderable clips found on disk, with their label files."""
    clips = []
    for scenario in SCENARIOS:
        gt = data / scenario / "ground_truth.json"
        for camera in FIXED_CAMERAS:
            video = output / scenario / f"{camera}.mp4"
            if video.exists() and gt.exists():
                clips.append(Clip(scenario, camera, video, gt))
    race = output / "race_pace" / "race_pace.mp4"
    if include_race_pace and race.exists():
        clips.append(Clip("race_pace", "tracking_showcase", race,
                          output / "race_pace" / "ground_truth.json",
                          fps=60.0, moving_camera=True))
    return clips


def read_frames(path, limit=None):
    """Decode a clip to BGR frames in order."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    idx = 0
    try:
        while limit is None or idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_all(path, limit=None):
    return np.stack([f for _, f in read_frames(path, limit)])


def static_background(path, sample=32):
    """Temporal median of evenly spaced frames: the empty circuit, cars removed.

    Used by the self-calibration, which must work from video alone and must not
    be confused by the very cars it is later asked to judge.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    picks = np.unique(np.linspace(0, max(total - 1, 0), sample).astype(int))
    frames = []
    want = set(int(p) for p in picks)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
