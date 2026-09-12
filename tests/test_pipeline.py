"""Checks on the parts of the stack that must be exactly right.

These run in seconds and need no trained weights: they pin the geometry against
the simulator's own exports, and they pin the judgement rules against cases
built by hand where the answer is known in advance.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vmax_vision import boundary, clips, track_model as tm  # noqa: E402
from vmax_vision.calib import Camera, apply_h, back_projection_error, track_frame_error  # noqa: E402
from vmax_vision.tracker import ByteTrack, iou_matrix  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "pass" if condition else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def test_geometry_matches_simulator():
    """The re-derived circuit model must reproduce the exported excess exactly."""
    worst_excess = worst_pose = 0.0
    for scenario in clips.SCENARIOS:
        gt = json.loads((clips.DATA / scenario / "ground_truth.json").read_text())
        for row in gt["frames"]:
            q = np.array([row["contact_points_world"][k] for k in tm.CONTACT_KEYS])
            truth = np.array([row["corner_excess_m"][k] for k in tm.CONTACT_KEYS])
            worst_excess = max(worst_excess, np.abs(tm.signed_excess(q) - truth).max())
            rebuilt = tm.car_contacts(np.array(row["world_position"]), row["heading_rad"])
            worst_pose = max(worst_pose, np.abs(rebuilt - q).max())
    check("per-corner excess matches the simulator", worst_excess < 1e-12,
          f"max {worst_excess:.2e} m")
    check("contact geometry rebuilds from pose", worst_pose < 1e-12, f"max {worst_pose:.2e} m")


def test_track_coordinates_round_trip():
    rng = np.random.default_rng(0)
    s = rng.uniform(tm.ENTRY_S, tm.EXIT_S, 20000)
    lat = rng.uniform(-12, 12, 20000)
    c, n = tm.centreline(s)
    p = c + lat[:, None] * n
    s2, lat2 = tm.track_coordinates(p)
    check("track coordinates round-trip", max(np.abs(s2 - s).max(), np.abs(lat2 - lat).max()) < 1e-10)


def test_homography_agrees_with_projection():
    worst = 0.0
    for cam in Camera.load_all().values():
        pts = np.array([[0.0, 0.0], [40.0, 40.0], [10.0, -6.9], [35.0, 25.0]])
        uv3d, _ = cam.project(np.column_stack([pts, np.zeros(len(pts))]))
        worst = max(worst, np.abs(cam.ground_to_pixel(pts) - uv3d).max())
        err = back_projection_error(cam.H, cam)
        check(f"{cam.name}: identity back-projection", err["max_ground_error_m"] < 1e-9)
        lat = track_frame_error(cam.H, cam)
        check(f"{cam.name}: identity lateral error", lat["max_abs_lateral_error_m"] < 1e-9)
    check("ground homography agrees with full projection", worst < 1e-9, f"max {worst:.2e} px")


def _synthetic_track(offsets, camera, noise=0.0, seed=0):
    """A car driven at given lateral offsets, projected through a real camera.

    Returns the detections a perfect detector would produce, and the exact
    per-frame margin that pose actually carries. The margin is *computed*, not
    assumed: on a curved track the front and rear contacts sit further from the
    centre than the car does, so a commanded offset is not the margin.
    """
    rng = np.random.default_rng(seed)
    s = 8.0 + 12.0 * np.arange(len(offsets)) / 24.0
    centre, normal = tm.centreline(s)
    pos = centre - normal * np.asarray(offsets)[:, None]
    velocity = np.gradient(pos, axis=0)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    frames, expected = {}, []
    for i in range(len(offsets)):
        contacts = tm.car_contacts(pos[i], heading[i])
        excess = tm.signed_excess(contacts)
        expected.append(float(excess.min()) if (excess > 0).all() else float(excess.min()))
        uv = camera.ground_to_pixel(contacts) + rng.normal(0, noise, (4, 2))
        frames[i] = {"score": 0.9, "box": [0, 0, 1, 1], "contacts_uv": uv.tolist()}
    return frames, np.array(expected)


def test_judgement_recovers_a_known_excursion():
    cam = Camera.load_all()["trackside"]
    n = 96
    offsets = np.full(n, 5.4)
    offsets[30:60] = 7.0 + 0.82 + 0.20
    frames, expected = _synthetic_track(offsets, cam)
    js = boundary.judge_clip(frames, cam.H_inv, smooth=False)
    events = boundary.find_events(js, min_frames=3)
    off_track = np.flatnonzero(expected > 0)
    check("one sustained excursion found", len(events) == 1, f"got {len(events)}")
    if events:
        ev = events[0]
        check("peak margin recovered", abs(ev.peak_margin_m - expected.max()) < 2e-3,
              f"{ev.peak_margin_m:.4f} m vs {expected.max():.4f} m")
        check("excursion window recovered",
              ev.start_frame == off_track[0] and ev.end_frame_inclusive == off_track[-1],
              f"{ev.start_frame}-{ev.end_frame_inclusive} vs {off_track[0]}-{off_track[-1]}")


def test_two_frame_blip_is_suppressed():
    cam = Camera.load_all()["trackside"]
    # Ramped, not stepped: a one-frame lateral jump would swing the heading and
    # skew the car, which is a different test from a brief excursion.
    offsets = np.full(96, 5.4)
    ramp = 5.4 + (7.0 + 0.82 + 0.30 - 5.4) * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, 13)))
    offsets[42:55] = ramp
    frames, expected = _synthetic_track(offsets, cam)
    off_track = int((expected > 0).sum())
    check("the blip is a brief excursion", 1 <= off_track <= 4, f"{off_track} frames off track")
    js = boundary.judge_clip(frames, cam.H_inv, smooth=False)
    check("a blip below the sustained threshold is not reported",
          len(boundary.find_events(js, min_frames=off_track + 1)) == 0)
    check("the same blip is reported when the rule allows it",
          len(boundary.find_events(js, min_frames=off_track)) == 1)


def test_rigid_fit_absorbs_contact_noise():
    cam = Camera.load_all()["trackside"]
    offsets = np.full(96, 7.0 + 0.82 + 0.15)
    clean_frames, expected = _synthetic_track(offsets, cam)
    clean = boundary.judge_clip(clean_frames, cam.H_inv, smooth=False)
    check("clean judgement is exact",
          max(abs(j.margin_m - e) for j, e in zip(clean, expected)) < 1e-6)
    noisy_frames, _ = _synthetic_track(offsets, cam, noise=1.0, seed=3)
    noisy = boundary.judge_clip(noisy_frames, cam.H_inv, smooth=False)
    raw_err = np.mean([abs(j.raw_excess.min() - e) for j, e in zip(noisy, expected)])
    fit_err = np.mean([abs(j.margin_m - e) for j, e in zip(noisy, expected)])
    check("rigid fit beats raw back-projection under noise", fit_err < raw_err,
          f"{fit_err:.4f} m vs {raw_err:.4f} m")


def test_tracker_keeps_identity_through_a_dropout():
    tracker = ByteTrack(min_hits=2)
    ids = []
    for i in range(30):
        box = [100 + 4 * i, 200, 160 + 4 * i, 240]
        # Frames 12-14 come back only as weak detections, which is exactly the
        # case ByteTrack's second association pass exists for.
        score = 0.2 if 12 <= i <= 14 else 0.9
        live = tracker.update([{"score": score, "box": box, "contacts_uv": None}], i)
        ids.append([t.track_id for t in live])
    stable = {i[0] for i in ids if i}
    check("identity survives a low-confidence stretch", len(stable) == 1, f"ids {stable}")


def test_iou_matrix():
    a = [[0, 0, 10, 10]]
    check("iou of identical boxes", abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-9)
    check("iou of disjoint boxes", iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0.0)
    check("iou of half overlap", abs(iou_matrix(a, [[5, 0, 15, 10]])[0, 0] - 1 / 3) < 1e-9)


def main():
    for fn in [test_geometry_matches_simulator, test_track_coordinates_round_trip,
               test_homography_agrees_with_projection, test_judgement_recovers_a_known_excursion,
               test_two_frame_blip_is_suppressed, test_rigid_fit_absorbs_contact_noise,
               test_tracker_keeps_identity_through_a_dropout, test_iou_matrix]:
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
