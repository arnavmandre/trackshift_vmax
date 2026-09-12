"""Conservative planar contact-footprint clearance under an explicit tyre model.

A grid covers each rectangular patch. Distance to the track centreline is
1-Lipschitz. Subtracting the grid covering radius bounds the true minimum from
below: a positive lower bound establishes separation under this model.
Not a model of deformation, raised kerbs or airborne wheels.
"""
import numpy as np
from . import track_model as tm


def clearance(contacts, heading, width_m=0.36, length_m=0.12, spacing_m=0.02, track_map=None):
    if min(width_m, length_m, spacing_m) <= 0:
        raise ValueError('patch dimensions and spacing must be positive')
    nx = max(2, int(np.ceil(length_m/spacing_m))+1)
    ny = max(2, int(np.ceil(width_m/spacing_m))+1)
    x = np.linspace(-length_m/2, length_m/2, nx)
    y = np.linspace(-width_m/2, width_m/2, ny)
    local = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    points = np.asarray(contacts)[:, None, :] + (local @ rot.T)[None, :, :]
    values = (track_map or tm).signed_excess(points)
    error = 0.5 * np.hypot(length_m/(nx-1), width_m/(ny-1))
    upper = values.min(axis=1)
    return upper-error, upper


def apply(result, width_m=0.36, length_m=0.12, track_map=None):
    """Replace point margins before incident extraction, retaining raw values."""
    for track in result['tracks'].values():
        for f in track['frames']:
            lo, hi = clearance(f['contacts_world'], f['heading_rad'], width_m, length_m, track_map=track_map)
            f['point_margin_m'] = f['margin_m']
            f['footprint_clearance_bounds_m'] = np.stack([lo, hi], axis=1).tolist()
            f['margin_m'] = float(lo.min())
            f['excess_m'] = lo.tolist()
            f['is_violation'] = bool((lo > 0).all())
        # Keep brief candidates. A known legal/uncertain patch splits the run.
        events, run, prev = [], [], None
        def flush():
            if not run:
                return
            peak = max(run, key=lambda f:f['margin_m'])
            events.append(dict(start_frame=run[0]['frame_idx'],end_frame_inclusive=run[-1]['frame_idx'],
                start_time_s=run[0]['time_s'],end_time_exclusive_s=run[-1]['time_s']+1/result['fps'],
                frame_count=len(run),peak_frame=peak['frame_idx'],peak_margin_m=peak['margin_m'],
                mean_score=float(np.mean([f['score'] for f in run])),confidence=0.0,
                confidence_kind='not estimated for footprint mode',driver_confidence=(track.get('attribution') or {}).get('confidence',0),
                incident_score=0.0,review_status='needs_review'))
        for f in sorted(track['frames'],key=lambda f:f['frame_idx']):
            if prev is not None and f['frame_idx']-prev > 1:
                flush();run=[]
            if f['is_violation']:
                run.append(f)
            else:
                flush();run=[]
            prev=f['frame_idx']
        flush()
        track['events']=events
    result['geometry_contract']='planar rectangular tyre footprints; conservative grid clearance'
    result['footprint_model']={'width_m':width_m,'length_m':length_m,'deformation_modelled':False}
    return result
