"""Assemble one compact JSON for the steward dashboard from the run artefacts."""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import numpy as np

from . import clips as clipmod

OUT = pathlib.Path("pipeline_out")

TARGETS = {"violation_0.05m": 0.05, "violation_0.15m": 0.15,
           "violation_0.30m": 0.30, "violation_0.60m": 0.60}


def _true_events(clip):
    gt = clip.ground_truth()
    return gt["events"]


def _trace(rows, scenario, camera, mode):
    sel = [r for r in rows if r["scenario"] == scenario and r["camera"] == camera
           and r["calibration_mode"] == mode]
    by_car = {}
    for r in sel:
        by_car.setdefault(r["car_id"], []).append(r)
    out = {}
    for car, items in by_car.items():
        items.sort(key=lambda r: r["frame_idx"])
        out[car] = {
            "frame": [r["frame_idx"] for r in items],
            "detected": [round(r["detected_margin_m"], 4) for r in items],
            "truth": [round(r["true_margin_m"], 4) for r in items],
            "detected_violation": [bool(r["detected_violation"]) for r in items],
            "true_violation": [bool(r["true_violation"]) for r in items],
            "score": [round(r["score"], 3) for r in items],
        }
    return out


def build(evaluation="pipeline_out/evaluation.json",
          calibration="pipeline_out/self_calibration.json",
          training="pipeline_out/training_log.json",
          baseline="pipeline_out/baseline_coco.json",
          out="pipeline_out/dashboard.json"):
    report = json.loads(pathlib.Path(evaluation).read_text())
    by_key = {c.key: c for c in clipmod.discover(include_race_pace=False)}

    cases = []
    graduated = []
    for name, entry in report["clips"].items():
        clip = by_key[f"{entry['scenario']}/{entry['camera']}"]
        truth_events = _true_events(clip)
        detected = [e for e in entry["events"]["detected_events"]]
        best = max(detected, key=lambda e: e["detected"]["confidence"], default=None)
        true_peak = max((v.get("peak_margin_m", float("-inf"))
                         for v in truth_events.values()), default=None)
        true_any = any(v.get("events") for v in truth_events.values())
        verdict = "offence" if best else ("no offence" if not true_any else "missed")
        if best and not best.get("matched"):
            verdict = "false alarm"
        case = {
            "id": name,
            "scenario": entry["scenario"],
            "camera": entry["camera"],
            "calibration_mode": entry["calibration_mode"],
            "detector": entry["detector"],
            "held_out": entry["held_out"],
            "held_out_scenario": entry["held_out_scenario"],
            "verdict": verdict,
            "confidence": best["detected"]["confidence"] if best else 0.0,
            "peak_margin_detected": best["detected"]["peak_margin_m"] if best else None,
            "peak_margin_true": true_peak if true_any else None,
            "event_detected": best["detected"] if best else None,
            "event_truth": best.get("truth") if best else None,
            "start_error_frames": best.get("start_error_frames") if best else None,
            "duration_error_frames": best.get("duration_error_frames") if best else None,
            "missed_events": entry["events"]["missed_events"],
            "correctly_suppressed_events": entry["events"].get("correctly_suppressed_events", []),
            "sustained_min_frames": entry["events"].get("sustained_min_frames", 3),
            "true_event_count": entry["events"]["true_event_count"],
            "detection": entry["detection"],
            "margin": entry["margin"],
            "attribution": entry["attribution"],
            "trace": _trace(report["margin_rows"], entry["scenario"], entry["camera"],
                            entry["calibration_mode"]),
        }
        cases.append(case)
        if entry["scenario"] in TARGETS and best:
            graduated.append({
                "scenario": entry["scenario"],
                "camera": entry["camera"],
                "calibration_mode": entry["calibration_mode"],
                "target_m": TARGETS[entry["scenario"]],
                "true_peak_m": best.get("truth", {}).get("peak_margin_m"),
                "detected_peak_m": best["detected"]["peak_margin_m"],
                "held_out": entry["held_out"],
            })

    calib = []
    cal_path = pathlib.Path(calibration)
    if cal_path.exists():
        for camera, entry in json.loads(cal_path.read_text()).items():
            calib.append({
                "camera": camera,
                "position_estimated_m": [round(v, 3) for v in entry["position_m"]],
                "position_true_m": entry["truth"]["position_m"],
                "position_error_m": entry["position_error_m"],
                "fov_estimated_deg": round(entry["vertical_fov_deg"], 3),
                "fov_true_deg": entry["truth"]["vertical_fov_deg"],
                "lateral_error_m": entry["track_frame_error"]["mean_abs_lateral_error_m"],
                "lateral_p95_m": entry["track_frame_error"]["p95_abs_lateral_error_m"],
                "along_track_gauge_m": entry["track_frame_error"]["along_track_gauge_m"],
                "ground_error_m": entry["ground_frame_error"]["mean_ground_error_m"],
                "reprojection_px": entry["ground_frame_error"]["mean_reprojection_error_px"],
                "paint_agreement": entry.get("paint_agreement"),
                "paint_agreement_at_truth": entry.get("paint_agreement_at_truth"),
                "failure_mode": entry.get("failure_mode"),
                "seconds": entry.get("seconds"),
                "camera_role": "solved from video",
            })

    train_log = []
    tp = pathlib.Path(training)
    if tp.exists():
        raw = json.loads(tp.read_text())
        step = max(1, len(raw) // 220)
        train_log = [{"step": r["step"], "loss": round(r["loss"], 3),
                      "contact": round(r["contact"], 4)} for r in raw[::step]]

    baseline_data = None
    bp = pathlib.Path(baseline)
    if bp.exists():
        baseline_data = json.loads(bp.read_text())

    peak_errors = [abs(g["detected_peak_m"] - g["target_m"]) for g in graduated
                   if g["calibration_mode"] == "surveyed"]
    payload = {
        "headline": {
            "peak_margin_median_error_m": float(np.median(peak_errors)) if peak_errors else None,
            "peak_margin_max_error_m": float(max(peak_errors)) if peak_errors else None,
            "graduated_points": len(peak_errors),
        },
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "held_out_by_scenario": clipmod.HELD_OUT_BY_SCENARIO,
            "held_out_scenarios": clipmod.HELD_OUT_SCENARIOS,
            "train_cameras": clipmod.TRAIN_CAMERAS,
            "held_out_clips": sum(1 for c in cases if c["held_out"]),
            "clip_count": len(cases),
        },
        "summary": report["summary"],
        "cases": sorted(cases, key=lambda c: (c["scenario"], c["camera"], c["calibration_mode"])),
        "graduated": graduated,
        "calibration": calib,
        "training": train_log,
        "baseline": baseline_data,
    }
    pathlib.Path(out).write_text(json.dumps(payload))
    return payload


if __name__ == "__main__":
    p = build()
    print(f"{len(p['cases'])} cases, {len(p['graduated'])} graduated points")
