"""Command line for the VMAX steward pipeline."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path("pipeline_out")


def cmd_selfcal(args):
    import time
    from . import clips, selfcalib
    from .calib import Camera, back_projection_error, track_frame_error
    cams = Camera.load_all()
    out = {}
    for name in clips.FIXED_CAMERAS:
        best = None
        for seed in args.seeds:
            t0 = time.time()
            bg = clips.static_background(clips.OUTPUT / "clean_lap" / f"{name}.mp4")
            est = selfcalib.estimate_homography(bg, seed=seed)
            est["seed"] = seed
            est["seconds"] = round(time.time() - t0, 1)
            print(f"  {name} seed {seed}: agreement {est['paint_agreement']:.3f}", flush=True)
            if best is None or est["paint_agreement"] > best["paint_agreement"]:
                best = est
        H = np.asarray(best.pop("homography"))
        best["homography"] = H.tolist()
        best["source_clip"] = f"clean_lap/{name}.mp4"
        best["ground_frame_error"] = back_projection_error(H, cams[name])
        best["track_frame_error"] = track_frame_error(H, cams[name])
        best["truth"] = {"position_m": cams[name].position.tolist(),
                         "look_at_m": cams[name].spec["look_at_m"],
                         "vertical_fov_deg": cams[name].spec["vertical_fov_deg"]}
        best["position_error_m"] = float(
            np.linalg.norm(np.array(best["position_m"]) - cams[name].position))
        # Score the true camera by the same measure, so a failure is attributed
        # to the search or to the objective rather than left ambiguous.
        best["paint_agreement_at_truth"] = selfcalib.paint_agreement(
            bg, cams[name].position, cams[name].spec["look_at_m"],
            cams[name].spec["vertical_fov_deg"])
        best["failure_mode"] = (
            None if best["position_error_m"] < 3.0
            else ("objective" if best["paint_agreement_at_truth"] <= best["paint_agreement"]
                  else "search"))
        out[name] = best
        print(f"{name}: position error {best['position_error_m']:.2f} m, "
              f"lateral {best['track_frame_error']['mean_abs_lateral_error_m']:.3f} m", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "self_calibration.json").write_text(json.dumps(out, indent=2))


def cmd_train(args):
    from .train import main as train_main
    argv = ["--iterations", str(args.iterations), "--batch", str(args.batch)]
    if args.resume:
        argv += ["--resume", args.resume]
    train_main(argv)


def _detector(args):
    if args.detector == "yolo-coco":
        from .detector import CocoYoloDetector
        return CocoYoloDetector(threshold=args.threshold)
    from .detector import VmaxDetector
    return VmaxDetector(weights=args.weights, threshold=args.threshold)


def cmd_run(args):
    from .pipeline import run_all
    detector = _detector(args)
    run_all(detector, mode=args.mode, scenarios=args.scenarios or None,
            cameras_wanted=args.cameras or None, out_dir=args.out)


def cmd_demo(args):
    """Judge one clip and print the verdict the way a steward would read it."""
    from . import clips as clipmod
    from .pipeline import run_clip
    by_key = {c.key: c for c in clipmod.discover(include_race_pace=False)}
    if args.clip not in by_key:
        print(f"unknown clip {args.clip!r}. available:")
        for key in sorted(by_key):
            print(f"  {key}")
        return 1
    clip = by_key[args.clip]
    print(f"reading {clip.video} ({clip.camera} camera)", flush=True)
    detector = _detector(args)
    result = run_clip(clip, detector, mode=args.mode)

    truth = clip.ground_truth()["events"]
    print()
    print(f"  clip            {clip.key}")
    print(f"  calibration     {result['calibration_label']}")
    print(f"  withheld from training  {'yes' if clip.held_out else 'no'}")
    print(f"  cars tracked    {len(result['tracks'])}")
    any_event = False
    for tid, track in sorted(result["tracks"].items()):
        car = (track.get("attribution") or {}).get("car_id") or f"track {tid}"
        rung = (track.get("attribution") or {}).get("rung", "-")
        for ev in track["events"]:
            any_event = True
            actual = truth.get(car, {}).get("events", [])
            print()
            print(f"  OFFENCE  {car}  (identified by {rung})")
            print(f"    frames        {ev['start_frame']}-{ev['end_frame_inclusive']}"
                  f"  ({ev['frame_count']} off track, {ev['start_time_s']:.2f}-{ev['end_time_exclusive_s']:.2f} s)")
            print(f"    peak margin   {ev['peak_margin_m']:+.3f} m beyond the white line")
            print(f"    detection     {ev['confidence'] * 100:.0f}% heuristic excursion score (uncalibrated)")
            print(f"    driver ID     {ev['driver_confidence'] * 100:.0f}% confident it was {car}")
            print(f"    INCIDENT      {ev['incident_score'] * 100:.0f}%"
                  "   (detection x driver ID -- a shaky identification drags it down)")
            if actual:
                a = actual[0]
                print(f"    ground truth  frames {a['start_frame']}-{a['end_frame_inclusive']}, "
                      f"peak {a['peak_margin_m']:+.3f} m")
                print(f"    error         {ev['peak_margin_m'] - a['peak_margin_m']:+.3f} m, "
                      f"{ev['start_frame'] - a['start_frame']:+d} frames on the start")
    if not any_event:
        print()
        print("  NO OFFENCE  no sustained excursion in this clip")
        for car, info in truth.items():
            for a in info.get("events", []):
                print(f"    note: ground truth has a {a['frame_count']}-frame excursion by {car} "
                      f"peaking at {a['peak_margin_m']:+.3f} m")
    print()
    return 0


def cmd_batch(args):
    """Judge many clips and print one row each: verdict, margin, truth, error.

    This is the demo view -- it shows the system working across a whole session
    rather than on one hand-picked clip, and puts the answer next to the truth
    on every line so nothing has to be taken on trust.
    """
    import time
    import numpy as np
    from . import clips as clipmod
    from .pipeline import run_clip

    pool = [c for c in clipmod.discover(include_race_pace=False)
            if (not args.cameras or c.camera in args.cameras)
            and (not args.scenarios or c.scenario in args.scenarios)]
    if not pool:
        print("no clips matched those filters")
        return 1
    detector = _detector(args)

    head = (f"{'CLIP':<34}{'VERDICT':<13}{'REPORTED':>10}{'TRUTH':>10}"
            f"{'ERROR':>9}{'DETECT':>8}{'DRIVER':>8}{'INCIDENT':>10}{'CAR':>9}   ")
    print()
    print(head)
    print("-" * (len(head) + 2))

    peak_errors, found, missed, false_alarms, suppressed, unnamed = [], 0, 0, 0, 0, 0
    named_right = named_total = 0
    started = time.time()
    for clip in pool:
        result = run_clip(clip, detector, mode=args.mode, cache=not args.no_cache)
        truth = clip.ground_truth()["events"]
        true_events = [(car, e) for car, info in truth.items() for e in info.get("events", [])]
        rows = []
        for tid, track in sorted(result["tracks"].items()):
            car = (track.get("attribution") or {}).get("car_id")
            actual = _nearest_truth_car(track, clip)
            if car and actual:
                named_total += 1
                named_right += int(car == actual)
            for ev in track["events"]:
                # Only a named car is matched against a truth event. An
                # unattributed track is shown as unnamed rather than quietly
                # borrowing the answer -- declining to name is the design, and
                # a steward needs to see which reports carry a name.
                match = next((e for c, e in true_events if car and c == car
                              and e["end_frame_inclusive"] >= ev["start_frame"]
                              and e["start_frame"] <= ev["end_frame_inclusive"]), None)
                rows.append((car, ev, match))

        reported = sorted(rows, key=lambda r: -r[1]["confidence"])
        for car, ev, match in reported:
            if match:
                found += 1
                err = ev["peak_margin_m"] - match["peak_margin_m"]
                peak_errors.append(abs(err))
                mark = "ok" if abs(err) < 0.10 else "!!"
                print(f"{clip.key:<34}{'OFFENCE':<13}{ev['peak_margin_m']:>+10.3f}"
                      f"{match['peak_margin_m']:>+10.3f}{err:>+9.3f}"
                      f"{ev['confidence']*100:>7.0f}%{ev['driver_confidence']*100:>7.0f}%"
                      f"{ev['incident_score']*100:>9.0f}%{car:>9}   {mark}")
            elif car is None:
                unnamed += 1
                print(f"{clip.key:<34}{'for review':<13}{ev['peak_margin_m']:>+10.3f}"
                      f"{'-':>10}{'-':>9}{ev['confidence']*100:>7.0f}%{'0%':>8}{'0%':>10}"
                      f"{'unnamed':>9}   ?   (no confident identification, escalated)")
            else:
                false_alarms += 1
                print(f"{clip.key:<34}{'FALSE ALARM':<13}{ev['peak_margin_m']:>+10.3f}"
                      f"{'-':>10}{'-':>9}{ev['confidence']*100:>7.0f}%"
                      f"{ev['driver_confidence']*100:>7.0f}%{ev['incident_score']*100:>9.0f}%"
                      f"{car:>9}   !!")
        for car, e in true_events:
            if any(m and m["start_frame"] == e["start_frame"] for _c, _ev, m in reported):
                continue
            if e["frame_count"] < result.get("sustained_min_frames", 3):
                suppressed += 1
                print(f"{clip.key:<34}{'blip ignored':<13}{'-':>10}{e['peak_margin_m']:>+10.3f}"
                      f"{'-':>9}{'-':>8}{'-':>8}{'-':>10}{car:>9}   ok  "
                      f"({e['frame_count']} frames, below the sustained threshold)")
            else:
                missed += 1
                print(f"{clip.key:<34}{'MISSED':<13}{'-':>10}{e['peak_margin_m']:>+10.3f}"
                      f"{'-':>9}{'-':>8}{'-':>8}{'-':>10}{car:>9}   !!")
        if not reported and not true_events:
            print(f"{clip.key:<34}{'no offence':<13}{'-':>10}{'-':>10}{'-':>9}"
                  f"{'-':>8}{'-':>8}{'-':>10}{'-':>9}   ok")

    print("-" * (len(head) + 2))
    total = found + missed
    print(f"{len(pool)} clips  |  {found}/{total} offences found  |  {false_alarms} false alarms"
          f"  |  {suppressed} blips correctly ignored  |  {unnamed} escalated unnamed")
    if peak_errors:
        print(f"peak-margin error: median {np.median(peak_errors)*100:.1f} cm, "
              f"worst {max(peak_errors)*100:.1f} cm")
    if named_total:
        print(f"car named correctly: {named_right}/{named_total}")
    print(f"{time.time()-started:.0f} s")
    print()
    return 0


def _nearest_truth_car(track, clip):
    """Which car this track actually followed, by nearest world position."""
    import numpy as np
    gt = clip.ground_truth()
    per_car = {}
    for row in gt["frames"]:
        per_car.setdefault(row["car_id"], {})[row["frame_idx"]] = row["world_position"]
    best, best_d = None, 1e9
    for car, frames in per_car.items():
        d = [float(np.linalg.norm(np.array(frames[f["frame_idx"]]) - np.array(f["position_m"])))
             for f in track["frames"][:40] if f["frame_idx"] in frames]
        if d and sum(d) / len(d) < best_d:
            best, best_d = car, sum(d) / len(d)
    return best if best_d < 4.0 else None


def cmd_evaluate(args):
    from .evaluate import score_run
    report = score_run(run_dir=args.runs, out=args.out)
    print(json.dumps(report["summary"], indent=2))


def cmd_overlay(args):
    from . import clips as clipmod
    from .overlay import render_clip
    from .pipeline import load_homography
    by_key = {c.key: c for c in clipmod.discover(include_race_pace=False)}
    for path in sorted(pathlib.Path(args.runs).glob("*.json")):
        result = json.loads(path.read_text())
        key = f"{result['scenario']}/{result['camera']}"
        if args.only and key not in args.only:
            continue
        H, _sigma, _label = load_homography(result["camera"], result["calibration_mode"])
        out = pathlib.Path(args.out) / f"{result['scenario']}__{result['camera']}__{result['calibration_mode']}.mp4"
        render_clip(by_key[key], result, out, H)
        print("wrote", out, flush=True)


def cmd_baseline(args):
    from .baseline import run
    run(weights=args.weights, stride=args.stride, conf=args.threshold)


def cmd_report(args):
    from .report import build
    payload = build()
    print(f"{len(payload['cases'])} cases written to pipeline_out/dashboard.json")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep the standalone, dependency-light workflows available from one entry point.
    extensions = {"generate":"generate", "blind":"blind", "review":"review", "import-video":"import_video"}
    if argv and argv[0] in extensions:
        import importlib
        return importlib.import_module("." + extensions[argv[0]], __package__).main(argv[1:])
    ap = argparse.ArgumentParser(prog="vmax-steward", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("selfcal", help="estimate each camera's homography from video alone")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 13])
    p.set_defaults(func=cmd_selfcal)

    p = sub.add_parser("train", help="train VMAX-Net on the held-in clips")
    p.add_argument("--iterations", type=int, default=900)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="run the pipeline over the clips")
    p.add_argument("--detector", choices=["vmaxnet", "yolo-coco"], default="vmaxnet")
    p.add_argument("--weights", default="pipeline_out/vmaxnet.pt")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--mode", choices=["surveyed", "self"], default="surveyed")
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--cameras", nargs="*", default=None)
    p.add_argument("--out", default="pipeline_out/runs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo", help="judge one clip and print the verdict")
    p.add_argument("--clip", default="violation_0.30m/trackside",
                   help="scenario/camera, e.g. violation_0.30m/trackside")
    p.add_argument("--detector", choices=["vmaxnet", "yolo-coco"], default="vmaxnet")
    p.add_argument("--weights", default="pipeline_out/vmaxnet.pt")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--mode", choices=["surveyed", "self"], default="surveyed")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("batch", help="judge many clips and print a verdict table")
    p.add_argument("--cameras", nargs="*", default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--detector", choices=["vmaxnet", "yolo-coco"], default="vmaxnet")
    p.add_argument("--weights", default="pipeline_out/vmaxnet.pt")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--mode", choices=["surveyed", "self"], default="surveyed")
    p.add_argument("--no-cache", action="store_true",
                   help="recompute detections instead of reusing the cache")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("evaluate", help="score pipeline output against the labels")
    p.add_argument("--runs", default="pipeline_out/runs")
    p.add_argument("--out", default="pipeline_out/evaluation.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("overlay", help="render steward review footage")
    p.add_argument("--runs", default="pipeline_out/runs")
    p.add_argument("--out", default="pipeline_out/overlays")
    p.add_argument("--only", nargs="*", default=None)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("baseline", help="zero-shot COCO YOLOv8-seg baseline")
    p.add_argument("--weights", default="yolov8m-seg.pt")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.02)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("report", help="assemble the dashboard JSON")
    p.set_defaults(func=cmd_report)

    ap.epilog = "Additional workflows: generate, blind, review, import-video (use COMMAND --help)."
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
