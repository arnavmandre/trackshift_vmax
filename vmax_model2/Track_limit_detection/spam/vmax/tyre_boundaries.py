"""Lower tyre boundaries for the smooth rigid envelope used by the v2 renderer.

FIA B1.8.6 requires remaining contact with the track, including its white line.
This model has no deformation: its road-touching tread is a transverse line,
not a finite contact patch. Full sidewall width is NOT used for the decision.
"""
import copy
import numpy as np
import geometry as g

TYRE_RADIUS_M = 0.36
TYRE_PROFILE = ((-.18,.27),(-.17,.32),(-.13,.352),(-.08,.36),
                (.08,.36),(.13,.352),(.17,.32),(.18,.27))
GROUND_HALF_WIDTH_M = max(abs(y) for y, radius in TYRE_PROFILE if radius == TYRE_RADIUS_M)
BOUNDARY_SPACING_M = .001
RULE = {
    'schema_version': 'vmax.tyre_boundaries.v1',
    'fia_reference': '2026 F1 Sporting Regulations B1.8.6',
    'fia_url': 'https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_06_-_2026-04-28.pdf',
    'target': 'ends of the road-touching tread line; lower tyre outline for context',
    'geometry': 'smooth rigid uncambered tyre envelope on flat z=0 road',
    'tyre_width_m': .36, 'tyre_radius_m': TYRE_RADIUS_M,
    'ground_tread_width_m': 2 * GROUND_HALF_WIDTH_M,
    'finite_contact_patch_modelled': False, 'deformation_modelled': False,
    'occlusion_modelled': False,
    'visibility_note': 'in_frame is geometric only; hidden boundaries remain labelled (amodal)',
    'white_line_outer_offset_m': 7.0, 'white_line_width_m': .10,
    'decision': 'all four complete ground tread lines outside the track; boundary uncertainty yields null',
    'distance_note': 'signed excess samples have a conservative Lipschitz discretisation bound',
    'scenario_name_note': 'names retain original centre-based trajectory targets, not new boundary margins',
    'mesh_note': 'uses the smooth envelope; polygonal wheel-spin tessellation can lift vertices by sub-mm amounts',
}


def project(points, camera):
    xyz = np.asarray(points, dtype=float)
    uv, depth = g.project(xyz, camera)
    width, height = camera['resolution']
    front = depth > camera.get('near_m', .2)
    valid = front & np.isfinite(uv).all(1)
    inside = valid & (uv[:,0] >= 0) & (uv[:,0] < width) & (uv[:,1] >= 0) & (uv[:,1] < height)
    return {
        'pixels': [p.tolist() if ok else None for p, ok in zip(uv, valid)],
        'in_frame': inside.tolist(),
        'occlusion': 'unknown',
    }


def wheel_boundary(row, key, camera):
    """Return lower tread profile with actual z values, and the z=0 tread endpoints."""
    center = np.asarray(row['contact_points_world'][key], dtype=float)
    heading = row['heading_rad'] + (row.get('visual_steering_rad', 0.) if key.startswith('front') else 0.)
    transverse = np.array([-np.sin(heading), np.cos(heading)])
    profile = np.array(TYRE_PROFILE)
    xy = center + profile[:,0,None] * transverse
    lower = np.column_stack([xy, TYRE_RADIUS_M - profile[:,1]])
    ends_xy = center + np.array([-GROUND_HALF_WIDTH_M, GROUND_HALF_WIDTH_M])[:,None] * transverse
    ends = np.column_stack([ends_xy, np.zeros(2)])
    count = int(np.ceil(2 * GROUND_HALF_WIDTH_M / BOUNDARY_SPACING_M)) + 1
    ts = np.linspace(0, 1, count)
    samples = ends_xy[0] + ts[:,None] * (ends_xy[1] - ends_xy[0])
    excess = g.distance(samples) - g.HALF
    index = int(excess.argmin())
    # Distance to a set is 1-Lipschitz. Every segment point is within half a sample step.
    error = float(np.linalg.norm(ends_xy[1] - ends_xy[0]) / (2 * (count - 1)))
    upper = float(excess[index])
    lower_bound = upper - error
    outside = True if lower_bound > 0 else (False if upper <= 0 else None)
    result = {
        'wheel_heading_rad': float(heading),
        'lower_outline_world_xyz': lower.tolist(),
        'ground_tread_endpoints_world_xyz': ends.tolist(),
        'ground_tread_endpoint_order': ['wheel_right', 'wheel_left'],
        'nearest_track_sample_world_xyz': [*samples[index].tolist(), 0.],
        'minimum_excess_m_interval': [lower_bound, upper],
        'sampling_error_bound_m': error,
        'is_fully_outside': outside,
    }
    if camera is not None:
        result['lower_outline_image'] = project(lower, camera)
        result['ground_tread_endpoints_image'] = project(ends, camera)
        result['nearest_track_sample_image'] = project([[*samples[index], 0.]], camera)
    return result


def annotate_row(row, camera=None):
    result = copy.deepcopy(row)
    if 'legacy_center_rule' not in result:
        result['legacy_center_rule'] = {k: copy.deepcopy(row[k]) for k in
            ['corner_excess_m', 'min_excess_m', 'max_excess_m', 'is_violation'] if k in row}
    camera = camera or row.get('camera')
    wheels = {key: wheel_boundary(row, key, camera) for key in g.KEYS}
    intervals = np.array([v['minimum_excess_m_interval'] for v in wheels.values()])
    lower, upper = intervals.min(0)
    verdict = True if lower > 0 else (False if upper <= 0 else None)
    # Remove ambiguous old centre-based metric names from the new schema.
    for key in ['corner_excess_m', 'min_excess_m', 'max_excess_m']:
        result.pop(key, None)
    result.update(tyre_boundaries=wheels, boundary_margin_m_interval=[float(lower), float(upper)],
                  is_violation=verdict,
                  boundary_status='outside' if verdict is True else ('within' if verdict is False else 'review'))
    return result


def event_summary(rows, fps):
    summary = {}
    for cid in sorted({r['car_id'] for r in rows}):
        selected = sorted((r for r in rows if r['car_id'] == cid), key=lambda r:r['frame_idx'])
        events, active = [], []
        def close():
            if active:
                events.append({'start_frame': active[0]['frame_idx'], 'end_frame_inclusive': active[-1]['frame_idx'],
                               'start_time_s': active[0]['t'], 'end_time_exclusive_s': active[-1]['t'] + 1/fps,
                               'frame_count': len(active),
                               'peak_boundary_margin_lower_m': max(r['boundary_margin_m_interval'][0] for r in active)})
                active.clear()
        previous = None
        for row in selected:
            if previous is not None and row['frame_idx'] != previous + 1:
                close()
            if row['is_violation'] is True: active.append(row)
            else: close()
            previous = row['frame_idx']
        close()
        summary[cid] = {'events': events, 'violation_frames': sum(r['is_violation'] is True for r in selected),
                        'review_frames': sum(r['is_violation'] is None for r in selected)}
    return summary


def track_boundary(camera):
    # The outer edges of both white lines are the limits. These samples include entry/exit straights.
    center, normal = g.center(np.linspace(-30, np.pi*20+30, 1001))
    result = {}
    for name, sign in [('left', 1), ('right', -1)]:
        world = np.column_stack([center + sign*g.HALF*normal, np.zeros(len(center))])
        result[name] = {'world_xyz': world.tolist(), 'image': project(world, camera)}
    return result


def annotate_document(data, camera=None):
    result = copy.deepcopy(data)
    camera = camera or data.get('camera')
    result['label_contract'] = RULE
    result['legacy_center_events'] = data.get('legacy_center_events', data.get('events'))
    result['frames'] = [annotate_row(row, camera) for row in data['frames']]
    result['events'] = event_summary(result['frames'], data['fps'])
    if camera:
        result['camera'] = camera
        result['track_boundary'] = track_boundary(camera)
    return result
