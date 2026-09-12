"""Versioned, explicit evidence uncertainty and blind-run provenance.

Intervals are model-based bounds, not calibrated confidence intervals. They
retain calibration/systematic error and do not gain certainty from frame count.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_signature(video, weights, settings, source_paths=()):
    return hashlib.sha256(json.dumps({
        'version': 3, 'video': sha256(video), 'weights': sha256(weights) if weights else None,
        'settings': settings, 'source': {str(p): sha256(p) for p in source_paths},
    }, sort_keys=True).encode()).hexdigest()


def frame_evidence(margin, sigma_m, systematic_bound_m=0.0):
    if not all(math.isfinite(x) for x in (margin, sigma_m, systematic_bound_m)):
        raise ValueError('evidence values must be finite')
    if sigma_m < 0 or systematic_bound_m < 0:
        raise ValueError('uncertainties must be nonnegative')
    radius = 1.96 * sigma_m + systematic_bound_m
    lo, hi = margin - radius, margin + radius
    return {'margin_interval_m': [lo, hi], 'interval_kind': 'model-based 1.96-sigma envelope; uncalibrated',
            'geometry_status': 'outside' if lo > 0 else ('inside' if hi < 0 else 'uncertain')}


def enrich(result, contact_sigma_m=0.10, systematic_bound_m=0.05):
    """Expose assumptions rather than claiming estimated percentages are probabilities."""
    if contact_sigma_m < 0 or systematic_bound_m < 0:
        raise ValueError('uncertainties must be nonnegative')
    total = int(result['frames'])
    observed = set()
    for track in result['tracks'].values():
        sig = math.hypot(contact_sigma_m, float(result['calibration_sigma_m']))
        for f in track['frames']:
            f.update(frame_evidence(f['margin_m'], sig, systematic_bound_m))
            observed.add(f['frame_idx'])
        by_idx = {f['frame_idx']: f for f in track['frames']}
        for ev in track['events']:
            members = [f for i, f in by_idx.items() if ev['start_frame'] <= i <= ev['end_frame_inclusive']]
            ev['evidence_coverage'] = len(members) / max(1, ev['end_frame_inclusive']-ev['start_frame']+1)
            peak = by_idx.get(ev['peak_frame'])
            ev['peak_frame_margin_interval_m'] = peak['margin_interval_m'] if peak else None
            ev['review_status'] = 'needs_review'
        if track['frames']:
            lo, hi = min(by_idx), max(by_idx)
            track['observation_coverage'] = len(by_idx) / (hi-lo+1)
        else:
            track['observation_coverage'] = 0
    result['evidence'] = {
        'contact_sigma_m_assumed': contact_sigma_m,
        'systematic_bound_m_assumed': systematic_bound_m,
        'frames_with_any_car_fraction': len(observed)/max(total, 1),
        'coverage_note': 'Any-car visibility is not per-car recall or proof of a clean clip.',
        'clip_status': 'review_candidates' if any(t['events'] for t in result['tracks'].values()) else 'no_candidates_observed',
        'confidence_kind': 'heuristic scores; probability calibration not yet measured',
    }
    return result
