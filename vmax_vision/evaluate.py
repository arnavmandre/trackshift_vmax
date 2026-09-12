"""Score the pipeline's own output against the exact labels.

Nothing in here feeds back into the pipeline. It answers four questions a
steward would ask before trusting the system:

  1. Does it find the cars, on a camera it was never trained on?
  2. Does it keep them apart, when there are two?
  3. How far is the margin it reports from the margin that was actually driven?
  4. When it declares an offence, is the timing and the confidence honest?
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import numpy as np

from . import clips as clipmod
from . import dataset as ds
from . import track_model as tm
from .calib import Camera


def _truth_frames(clip, cameras):
    """Per-frame truth: image boxes, contacts and the exact margin, per car."""
    labels = ds.frame_labels(clip, cameras)
    out = {}
    for row in labels:
        out[row["frame_idx"]] = {
            inst["car_id"]: {
                "box": inst["box"],
                "contacts_uv": np.array(inst["contacts_uv"], float),
                "world_position": np.array(inst["world_position"], float),
                "max_excess_m": inst["max_excess_m"],
            } for inst in row["instances"]
        }
    return out


def _truth_margins(clip):
    """Exact per-frame margin (min corner excess) and events for each car."""
    gt = clip.ground_truth()
    margins = defaultdict(dict)
    for row in gt["frames"]:
        margins[row["car_id"]][row["frame_idx"]] = {
            "margin_m": float(row["min_excess_m"]),
            "is_violation": bool(row["is_violation"]),
            "position": np.array(row["world_position"], float),
        }
    return margins, gt["events"]


def detection_metrics(result, clip, cameras, iou_thresh=0.5):
    """Frame-level precision/recall plus contact-point and centre accuracy."""
    truth = _truth_frames(clip, cameras)
    tp = fp = fn = 0
    centre_err, contact_err = [], []
    per_frame_dets = defaultdict(list)
    for tid, track in result["tracks"].items():
        for f in track["frames"]:
            per_frame_dets[f["frame_idx"]].append((tid, f))

    detections_per_frame = result["detections_per_frame"]
    for idx in range(len(detections_per_frame)):
        gt_boxes = truth.get(idx, {})
        got = per_frame_dets.get(idx, [])
        if not gt_boxes:
            fp += len(got)
            continue
        # Match on world position: after back-projection this is the quantity the
        # rest of the system uses, and it avoids rewarding a loose box.
        gt_ids = list(gt_boxes)
        gt_pos = np.array([gt_boxes[k]["world_position"] for k in gt_ids])
        used = set()
        for tid, f in got:
            pos = np.array(f["position_m"])
            d = np.linalg.norm(gt_pos - pos, axis=1)
            j = int(d.argmin())
            if d[j] <= 3.0 and j not in used:
                used.add(j)
                tp += 1
                centre_err.append(float(d[j]))
            else:
                fp += 1
        fn += len(gt_ids) - len(used)

    # Contact-point error in image pixels, matched frame by frame.
    for idx, gt_boxes in truth.items():
        got = per_frame_dets.get(idx, [])
        if not got or not gt_boxes:
            continue
        gt_ids = list(gt_boxes)
        gt_pos = np.array([gt_boxes[k]["world_position"] for k in gt_ids])
        for tid, f in got:
            pos = np.array(f["position_m"])
            j = int(np.linalg.norm(gt_pos - pos, axis=1).argmin())
            got_world = np.array(f["contacts_world"])
            true_world = tm.car_contacts(gt_pos[j],
                                         _heading_of(clip, gt_ids[j], idx))
            contact_err.append(float(np.linalg.norm(got_world - true_world, axis=1).mean()))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "mean_position_error_m": float(np.mean(centre_err)) if centre_err else None,
        "p95_position_error_m": float(np.percentile(centre_err, 95)) if centre_err else None,
        "mean_contact_error_m": float(np.mean(contact_err)) if contact_err else None,
        "p95_contact_error_m": float(np.percentile(contact_err, 95)) if contact_err else None,
    }


_HEADING_CACHE = {}


def _heading_of(clip, car_id, frame_idx):
    key = clip.key
    if key not in _HEADING_CACHE:
        gt = clip.ground_truth()
        _HEADING_CACHE[key] = {(r["car_id"], r["frame_idx"]): r["heading_rad"] for r in gt["frames"]}
    return _HEADING_CACHE[key].get((car_id, frame_idx), 0.0)


def margin_metrics(result, clip):
    """Detected margin against the exact margin, frame by frame.

    This is the number the whole system exists to produce, so it is compared
    directly rather than through any derived rate.
    """
    truth, _events = _truth_margins(clip)
    rows = []
    for tid, track in result["tracks"].items():
        car = (track.get("attribution") or {}).get("car_id")
        if car is None or car not in truth:
            # Fall back to nearest-truth matching so an unattributed track still
            # contributes a margin comparison.
            car = _nearest_car(track, truth)
        if car is None:
            continue
        for f in track["frames"]:
            t = truth[car].get(f["frame_idx"])
            if t is None:
                continue
            rows.append({
                "frame_idx": f["frame_idx"],
                "car_id": car,
                "group": str(tid),
                "detected_margin_m": f["margin_m"],
                "true_margin_m": t["margin_m"],
                "detected_violation": f["is_violation"],
                "true_violation": t["is_violation"],
                "score": f["score"],
            })
    if not rows:
        return {"frames": 0}, []
    err = np.array([r["detected_margin_m"] - r["true_margin_m"] for r in rows])
    tp = sum(r["detected_violation"] and r["true_violation"] for r in rows)
    fp = sum(r["detected_violation"] and not r["true_violation"] for r in rows)
    fn = sum((not r["detected_violation"]) and r["true_violation"] for r in rows)
    tn = sum((not r["detected_violation"]) and not r["true_violation"] for r in rows)
    return {
        "frames": len(rows),
        "margin_bias_m": float(err.mean()),
        "margin_mae_m": float(np.abs(err).mean()),
        "margin_rmse_m": float(np.sqrt((err ** 2).mean())),
        "margin_p95_abs_m": float(np.percentile(np.abs(err), 95)),
        "frame_true_positive": tp, "frame_false_positive": fp,
        "frame_false_negative": fn, "frame_true_negative": tn,
        "frame_accuracy": (tp + tn) / max(len(rows), 1),
    }, rows


def _nearest_car(track, truth):
    best, best_d = None, 1e9
    for car, frames in truth.items():
        ds_ = []
        for f in track["frames"][:40]:
            t = frames.get(f["frame_idx"])
            if t is not None:
                ds_.append(np.linalg.norm(t["position"] - np.array(f["position_m"])))
        if ds_ and np.mean(ds_) < best_d:
            best, best_d = car, float(np.mean(ds_))
    return best if best_d < 4.0 else None


def event_metrics(result, clip):
    """Event-level agreement: did we call the same offences, at the same time?

    A truth event shorter than the sustained-frame threshold is separated out
    rather than counted as a miss: declining to report it is the rule working,
    which is the whole reason ``sustained_vs_blip`` exists.
    """
    min_frames = result.get("sustained_min_frames", 3)
    truth, events = _truth_margins(clip)
    rows = []
    matched_truth = set()
    for tid, track in result["tracks"].items():
        car = (track.get("attribution") or {}).get("car_id") or _nearest_car(track, truth)
        true_events = events.get(car, {}).get("events", []) if car else []
        for ev in track["events"]:
            best, best_overlap = None, 0
            for k, te in enumerate(true_events):
                overlap = (min(ev["end_frame_inclusive"], te["end_frame_inclusive"])
                           - max(ev["start_frame"], te["start_frame"]) + 1)
                if overlap > best_overlap:
                    best, best_overlap = k, overlap
            row = {
                "group": str(tid), "car_id": car,
                "detected": {k: ev[k] for k in
                             ("start_frame", "end_frame_inclusive", "frame_count",
                              "peak_margin_m", "confidence")},
                "matched": best is not None,
            }
            if best is not None:
                te = true_events[best]
                matched_truth.add((car, best))
                row["truth"] = te
                row["start_error_frames"] = ev["start_frame"] - te["start_frame"]
                row["end_error_frames"] = ev["end_frame_inclusive"] - te["end_frame_inclusive"]
                row["duration_error_frames"] = ev["frame_count"] - te["frame_count"]
                row["peak_margin_error_m"] = ev["peak_margin_m"] - te["peak_margin_m"]
            rows.append(row)
    missed = []
    for car, info in events.items():
        for k, te in enumerate(info.get("events", [])):
            if (car, k) not in matched_truth:
                missed.append({"car_id": car, "truth": te,
                               "below_sustained_threshold": te["frame_count"] < min_frames})
    return {
        "detected_events": rows,
        "missed_events": [m for m in missed if not m["below_sustained_threshold"]],
        "correctly_suppressed_events": [m for m in missed if m["below_sustained_threshold"]],
        "sustained_min_frames": min_frames,
        "true_event_count": sum(len(v.get("events", [])) for v in events.values()),
    }


def attribution_metrics(result, clip):
    """Was the offence pinned on the right car?"""
    truth, _events = _truth_margins(clip)
    rows = []
    for tid, track in result["tracks"].items():
        attribution = track.get("attribution") or {}
        actual = _nearest_car(track, truth)
        rows.append({
            "group": str(tid),
            "claimed": attribution.get("car_id"),
            "actual": actual,
            "rung": attribution.get("rung"),
            "confidence": attribution.get("confidence"),
            "correct": attribution.get("car_id") == actual and actual is not None,
        })
    decided = [r for r in rows if r["claimed"] is not None and r["actual"] is not None]
    return {
        "tracks": rows,
        "attributed": len(decided),
        "correct": sum(r["correct"] for r in decided),
        "accuracy": (sum(r["correct"] for r in decided) / len(decided)) if decided else None,
    }


def score_run(run_dir="pipeline_out/runs", out="pipeline_out/evaluation.json",
              verbose=True):
    """Score every pipeline result on disk and write one combined report."""
    run_dir = pathlib.Path(run_dir)
    cameras = Camera.load_all()
    by_key = {c.key: c for c in clipmod.discover(include_race_pace=False)}
    report = {"clips": {}, "margin_rows": [], "modes": {}}
    for path in sorted(run_dir.glob("*.json")):
        result = json.loads(path.read_text())
        clip = by_key[f"{result['scenario']}/{result['camera']}"]
        margins, rows = margin_metrics(result, clip)
        entry = {
            "scenario": result["scenario"],
            "camera": result["camera"],
            "calibration_mode": result["calibration_mode"],
            "detector": result["detector"],
            "held_out": clipmod.is_held_out(result["scenario"], result["camera"]),
            "held_out_scenario": result["scenario"] in clipmod.HELD_OUT_SCENARIOS,
            "detection": detection_metrics(result, clip, cameras),
            "margin": margins,
            "events": event_metrics(result, clip),
            "attribution": attribution_metrics(result, clip),
        }
        report["clips"][path.stem] = entry
        for r in rows:
            r.update(scenario=result["scenario"], camera=result["camera"],
                     calibration_mode=result["calibration_mode"])
        report["margin_rows"].extend(rows)
        if verbose:
            d, m = entry["detection"], entry["margin"]
            print(f"  {path.stem:52s} R{d['recall']:.3f} P{d['precision']:.3f} "
                  f"MAE {m.get('margin_mae_m', float('nan')):.4f} m", flush=True)
    report["summary"] = summarise(report)
    pathlib.Path(out).write_text(json.dumps(report, indent=1))
    return report


def summarise(report):
    """Headline numbers, split by the axes that decide whether to believe them."""
    groups = defaultdict(list)
    for entry in report["clips"].values():
        mode = entry["calibration_mode"]
        held = "held_out" if entry["held_out"] else "trained_on"
        for key in (f"mode:{mode}",
                    f"camera:{entry['camera']}",
                    held,
                    # The headline axis: withheld footage judged the way an
                    # installed system would judge it, from a surveyed camera.
                    f"{mode}:{held}",
                    "held_out_scenario" if entry["held_out_scenario"] else "trained_scenario"):
            groups[key].append(entry)
    out = {}
    for key, entries in groups.items():
        margins = [e["margin"] for e in entries
                   if e["margin"].get("frames") and np.isfinite(e["margin"].get("margin_mae_m", np.nan))]
        det = [e["detection"] for e in entries]
        out[key] = {
            "clips": len(entries),
            "event_clips": sum(1 for e in entries if e["events"]["true_event_count"]),
            "events_found": sum(len([r for r in e["events"]["detected_events"] if r["matched"]])
                                for e in entries),
            "events_missed": sum(len(e["events"]["missed_events"]) for e in entries),
            "false_alarms": sum(len([r for r in e["events"]["detected_events"] if not r["matched"]])
                                for e in entries),
            "attribution_correct": sum(e["attribution"]["correct"] for e in entries),
            "attribution_decided": sum(e["attribution"]["attributed"] for e in entries),
            "recall": float(np.mean([d["recall"] for d in det])),
            "precision": float(np.mean([d["precision"] for d in det])),
            "margin_mae_m": float(np.mean([m["margin_mae_m"] for m in margins])) if margins else None,
            "margin_rmse_m": float(np.mean([m["margin_rmse_m"] for m in margins])) if margins else None,
            "frame_accuracy": float(np.mean([m["frame_accuracy"] for m in margins])) if margins else None,
        }
    return out
