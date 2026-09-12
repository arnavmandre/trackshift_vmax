"""Camera model, ground homography and back-projection error metrics."""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import track_model as tm


def look_at_rotation(position, target):
    """World-to-camera rotation whose rows are (right, down, forward)."""
    position = np.asarray(position, float)
    fwd = np.asarray(target, float) - position
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.array([right, down, fwd])


def intrinsics(vertical_fov_deg, width, height):
    focal = height / 2 / np.tan(np.deg2rad(vertical_fov_deg) / 2)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def homography_from_pose(position, target, vertical_fov_deg, width, height):
    """Ground-plane (z=0) homography for a look-at pinhole camera."""
    rot = look_at_rotation(position, target)
    K = intrinsics(vertical_fov_deg, width, height)
    H = K @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ np.asarray(position, float)])
    return H / H[2, 2]


def apply_h(H, xy):
    """Map ground metres to pixels (or the inverse, with an inverted H)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    h = np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(H, float).T
    return h[:, :2] / h[:, 2:3]


class Camera:
    """A fixed camera as the pipeline sees it: intrinsics, pose and ground plane."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.width, self.height = spec["resolution"]
        self.fps = spec.get("fps", 24)
        self.position = np.array(spec["position_m"], float)
        self.K = np.array(spec["K"], float)
        self.R = np.array(spec["R_world_to_camera"], float)
        self.H = np.array(spec["ground_plane_homography"], float)
        self.H_inv = np.linalg.inv(self.H)
        self.spec = spec

    @classmethod
    def load_all(cls, path="output/calibration.json"):
        specs = json.loads(pathlib.Path(path).read_text())
        return {s["name"]: cls(s) for s in specs}

    def project(self, points3d):
        """Full 3-D projection. Returns pixels and camera-frame depth."""
        pts = np.atleast_2d(np.asarray(points3d, float))
        cam = (pts - self.position) @ self.R.T
        h = cam @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = h[:, :2] / h[:, 2:3]
        return uv, cam[:, 2]

    def ground_to_pixel(self, xy):
        return apply_h(self.H, xy)

    def pixel_to_ground(self, uv):
        return apply_h(self.H_inv, uv)

    def in_frame(self, uv, margin=0):
        uv = np.atleast_2d(uv)
        return ((uv[:, 0] >= -margin) & (uv[:, 0] < self.width + margin)
                & (uv[:, 1] >= -margin) & (uv[:, 1] < self.height + margin))


def evaluation_grid(camera, step=0.6):
    """Ground points on the racing surface that this camera actually sees.

    Homography error is only meaningful where the camera has coverage, so the
    grid is the visible part of the track surface plus its run-off shoulder.
    """
    s = np.arange(tm.ENTRY_S, tm.EXIT_S, step)
    offsets = np.arange(-9.0, 9.0 + 1e-9, step)
    c, n = tm.centreline(s)
    pts = (c[:, None, :] + offsets[None, :, None] * n[:, None, :]).reshape(-1, 2)
    uv, depth = camera.project(np.column_stack([pts, np.zeros(len(pts))]))
    keep = (depth > 0.5) & camera.in_frame(uv)
    return pts[keep]


def back_projection_error(H_est, camera, step=0.6):
    """Metre error of an estimated homography, measured the way it is used.

    Every visible ground point is projected to pixels with the true camera and
    back-projected to metres with the estimate; the residual is what a boundary
    judgement made through ``H_est`` would inherit.
    """
    pts = evaluation_grid(camera, step)
    uv = camera.ground_to_pixel(pts)
    est = apply_h(np.linalg.inv(np.asarray(H_est, float)), uv)
    err = np.linalg.norm(est - pts, axis=1)
    fwd = np.linalg.norm(apply_h(H_est, pts) - uv, axis=1)
    return {
        "samples": int(len(pts)),
        "mean_ground_error_m": float(err.mean()),
        "median_ground_error_m": float(np.median(err)),
        "p95_ground_error_m": float(np.percentile(err, 95)),
        "max_ground_error_m": float(err.max()),
        "mean_reprojection_error_px": float(fwd.mean()),
        "max_reprojection_error_px": float(fwd.max()),
    }


def track_frame_error(H_est, camera, step=0.6):
    """Homography error resolved into the coordinates a verdict depends on.

    A corner of constant radius is nearly unobservable in rotation about its own
    centre: a camera solution rotated along the arc reproduces the same picture
    and the same lateral geometry. That rotation shows up as a large ground-frame
    error while changing no track-limit margin at all, so the two are reported
    separately -- lateral error is the one that propagates into a decision.
    """
    pts = evaluation_grid(camera, step)
    uv = camera.ground_to_pixel(pts)
    est = apply_h(np.linalg.inv(np.asarray(H_est, float)), uv)
    s_true, lat_true = tm.track_coordinates(pts)
    s_est, lat_est = tm.track_coordinates(est)
    d_lat = lat_est - lat_true
    d_s = s_est - s_true
    gauge = float(np.median(d_s))
    return {
        "samples": int(len(pts)),
        "mean_abs_lateral_error_m": float(np.abs(d_lat).mean()),
        "p95_abs_lateral_error_m": float(np.percentile(np.abs(d_lat), 95)),
        "max_abs_lateral_error_m": float(np.abs(d_lat).max()),
        "along_track_gauge_m": gauge,
        "mean_abs_along_track_error_m": float(np.abs(d_s - gauge).mean()),
        "p95_abs_along_track_error_m": float(np.percentile(np.abs(d_s - gauge), 95)),
    }
