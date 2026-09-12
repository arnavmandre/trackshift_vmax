"""Training labels derived from the simulator's own exports.

The renderer knows exactly where each car is; the detector does not. Labels are
produced by projecting the exported world pose through the exported camera, so
the network is trained against the same geometry the evaluation scores it on.
Only clips on the training side of the split in :mod:`vmax_vision.clips` are
ever turned into labels.
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

from . import clips as clipmod
from . import track_model as tm
from .calib import Camera

# Car body extent in local metres (x forward, y left, z up), used to draw the
# box and the instance mask. Wider than the contact rectangle because bodywork
# and wings overhang the tyres -- which is exactly why the contacts, not the
# silhouette, are what the geometry head predicts.
BODY_HULL = np.array([
    [2.55, 0.95, 0.02], [2.55, -0.95, 0.02], [-2.75, 1.05, 0.02], [-2.75, -1.05, 0.02],
    [2.55, 0.95, 0.35], [2.55, -0.95, 0.35], [-2.75, 1.05, 1.10], [-2.75, -1.05, 1.10],
    [1.30, 0.90, 0.75], [1.30, -0.90, 0.75], [-0.60, 0.90, 1.05], [-0.60, -0.90, 1.05],
    [1.80, 1.05, 0.72], [1.80, -1.05, 0.72], [-1.80, 1.05, 0.72], [-1.80, -1.05, 0.72],
])

CAR_COLOURS = {"car_1": (218, 27, 39), "car_2": (11, 141, 155)}  # RGB, as rendered


def camera_for(clip, frame_row, cameras=None):
    """The camera that shot a given frame, fixed or per-frame for a moving rig."""
    if clip.moving_camera:
        return Camera(frame_row["camera"])
    cameras = cameras or Camera.load_all()
    return cameras[clip.camera]


def world_to_local(position, heading, local_points):
    """Car-local metres to world metres for a pose on the ground plane."""
    cos, sin = np.cos(heading), np.sin(heading)
    rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return local_points @ rot.T + np.array([position[0], position[1], 0.0])


def frame_labels(clip, cameras=None):
    """Per-frame instance labels: box, contacts, hull polygon and truth margin."""
    gt = clip.ground_truth()
    cameras = cameras or Camera.load_all()
    by_frame = {}
    for row in gt["frames"]:
        by_frame.setdefault(row["frame_idx"], []).append(row)

    out = []
    for idx in sorted(by_frame):
        rows = by_frame[idx]
        cam = camera_for(clip, rows[0], cameras)
        instances = []
        for row in rows:
            pos = np.array(row["world_position"], float)
            heading = float(row["heading_rad"])
            contacts_world = np.array([row["contact_points_world"][k] for k in tm.CONTACT_KEYS])
            contacts_uv, contacts_z = cam.project(np.column_stack([contacts_world, np.zeros(4)]))
            hull_world = world_to_local(pos, heading, BODY_HULL)
            hull_uv, hull_z = cam.project(hull_world)
            if not (np.all(hull_z > 0.5) and np.all(contacts_z > 0.5)):
                continue
            x0, y0 = hull_uv.min(axis=0)
            x1, y1 = hull_uv.max(axis=0)
            if x1 < 0 or y1 < 0 or x0 > cam.width or y0 > cam.height:
                continue
            instances.append({
                "car_id": row["car_id"],
                "box": [float(x0), float(y0), float(x1), float(y1)],
                "contacts_uv": contacts_uv.tolist(),
                "hull_uv": hull_uv.tolist(),
                "world_position": pos.tolist(),
                "heading_rad": heading,
                "max_excess_m": float(row["max_excess_m"]),
                "is_violation": bool(row["is_violation"]),
                "speed_mps": float(row.get("speed_mps", 0.0)),
            })
        out.append({"frame_idx": idx, "camera": cam.name, "instances": instances})
    return out


def instance_mask(hull_uv, width, height):
    """Filled convex silhouette of one car, used as the segmentation target."""
    mask = np.zeros((height, width), np.uint8)
    pts = np.asarray(hull_uv, np.float32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 1)
    return mask


def build(split="train", cameras=None, verbose=False):
    """Decode the training clips once and return frames with their labels."""
    cameras = cameras or Camera.load_all()
    samples = []
    for clip in clipmod.discover():
        if (split == "train") != clip.is_training():
            continue
        labels = frame_labels(clip, cameras)
        by_idx = {row["frame_idx"]: row for row in labels}
        for idx, frame in clip.frames():
            row = by_idx.get(idx)
            if row is None or not row["instances"]:
                continue
            samples.append({"image": frame, "clip": clip.key, "frame_idx": idx,
                            "camera": row["camera"], "instances": row["instances"]})
        if verbose:
            print(f"  {clip.key}: {len(labels)} labelled frames", flush=True)
    return samples


def build_manifest(manifest_path, labels_path, max_frames=512, frame_stride=4, seed=0):
    """Train only on explicitly assigned scene families, with bounded decoding.

    The labels file is training input only. Annotation hulls remain approximate;
    this loader does not claim pixel-perfect visible instance masks.
    """
    from .blind import validate_manifest, resolve
    manifest = validate_manifest(manifest_path)
    labels = json.loads(pathlib.Path(labels_path).read_text())["clips"]
    pool = [c for c in manifest["clips"] if c["split"] == "train"]
    if not pool or max_frames < 1 or frame_stride < 1:
        raise ValueError("training split must be nonempty; frame limits must be positive")
    candidates = [(c, i) for c in pool for i in range(0, c["frames"], frame_stride)]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(candidates), min(max_frames, len(candidates)), replace=False)
    wanted = {}
    for ix in picks:
        c, i = candidates[ix]
        wanted.setdefault(c["id"], set()).add(i)
    samples = []
    for c in pool:
        if c["id"] not in wanted:
            continue
        truth = labels[c["id"]]
        if truth["video_sha256"] != c["video_sha256"]:
            raise ValueError("training label/video mismatch")
        cam = Camera(c["camera_spec"])
        rows = {}
        for row in truth["frames"]:
            rows.setdefault(row["frame_idx"], []).append(row)
        video = resolve(pathlib.Path(manifest_path).parent, c["video"])
        for idx, image in clipmod.read_frames(video):
            if idx not in wanted[c["id"]]:
                continue
            instances = []
            for row in rows.get(idx, []):
                points = np.asarray(row["contacts_world"])
                uv, depth = cam.project(np.column_stack([points, np.zeros(4)]))
                hull, hull_depth = cam.project(world_to_local(row["world_position"],row["heading_rad"],BODY_HULL))
                if np.min(depth) <= .5 or np.min(hull_depth) <= .5:
                    continue
                box = np.r_[hull.min(axis=0),hull.max(axis=0)]
                if box[2]<0 or box[3]<0 or box[0]>=image.shape[1] or box[1]>=image.shape[0]:
                    continue
                instances.append({"car_id":row["car_id"],"box":box.tolist(),"contacts_uv":uv.tolist(),"hull_uv":hull.tolist()})
            if instances:
                samples.append({"image":image,"clip":c["id"],"frame_idx":idx,"camera":c["id"],"instances":instances})
    if not samples:
        raise ValueError("no usable training frames; inspect camera coverage")
    return samples
