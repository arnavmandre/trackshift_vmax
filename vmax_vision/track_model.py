"""Analytic circuit model.

This is the circuit map a steward's system is entitled to know: centreline
geometry, track width and the white-line boundary. It is deliberately a
re-derivation rather than an import of ``geometry.py`` so the judgement stack
never borrows the simulator's per-frame state. ``test_track_model`` checks it
against the simulator's exported excess values to 1e-12 m.
"""
from __future__ import annotations

import numpy as np

RADIUS = 40.0          # centreline arc radius, metres
HALF_WIDTH = 7.0       # centreline to outer edge of the white line
LINE_WIDTH = 0.10      # the stripe runs inward from 7.00 m to 6.90 m
STRIPE_MID = HALF_WIDTH - LINE_WIDTH / 2
ARC_LEN = np.pi * RADIUS / 2       # quarter circle, s in [0, 20*pi]
ENTRY_S = -30.0
EXIT_S = ARC_LEN + 30.0

TYRE_RADIUS = 0.36
# Ideal contact centres in car-local metres (x forward, y left).
CONTACT_LOCAL = np.array([[1.8, 0.82], [1.8, -0.82], [-1.8, 0.82], [-1.8, -0.82]])
CONTACT_KEYS = ["front_left", "front_right", "rear_left", "rear_right"]


def centreline(s):
    """Centreline point and left-hand unit normal at arc length ``s``."""
    s = np.asarray(s, float)
    a = np.clip(s, 0.0, ARC_LEN) / RADIUS
    p = np.stack([RADIUS * np.sin(a), RADIUS * (1 - np.cos(a))], axis=-1)
    p[..., 0] += np.minimum(s, 0.0)
    p[..., 1] += np.maximum(s - ARC_LEN, 0.0)
    normal = np.stack([-np.sin(a), np.cos(a)], axis=-1)
    return p, normal


def distance_to_centreline(p):
    """Exact closest distance from ``p`` to the finite centreline polyline/arc."""
    p = np.asarray(p, float)
    x, y = p[..., 0], p[..., 1]
    entry = np.hypot(x - np.clip(x, ENTRY_S, 0.0), y)
    exit_ = np.hypot(x - RADIUS, y - np.clip(y, RADIUS, RADIUS + 30.0))
    a = np.clip(np.arctan2(x, RADIUS - y), 0.0, np.pi / 2)
    arc = np.hypot(x - RADIUS * np.sin(a), y - (RADIUS - RADIUS * np.cos(a)))
    return np.minimum(np.minimum(entry, exit_), arc)


def signed_excess(p):
    """Distance beyond the outer edge of the white line. Positive means outside."""
    return distance_to_centreline(p) - HALF_WIDTH


def boundary_polyline(side=1, n=1400, s_lo=ENTRY_S, s_hi=EXIT_S, offset=HALF_WIDTH):
    """World-space polyline of one track-limit line."""
    s = np.linspace(s_lo, s_hi, n)
    c, nrm = centreline(s)
    return c + side * nrm * offset


def car_contacts(position, heading):
    """The four ideal tyre contact points for a car pose, in world metres."""
    position = np.asarray(position, float)
    heading = np.asarray(heading, float)
    fwd = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    left = np.stack([-fwd[..., 1], fwd[..., 0]], axis=-1)
    return (position[..., None, :]
            + CONTACT_LOCAL[:, 0, None] * fwd[..., None, :]
            + CONTACT_LOCAL[:, 1, None] * left[..., None, :])


def pose_from_contacts(contacts):
    """Least-squares car centre and heading from four (possibly noisy) contacts.

    Solves the similarity-free rigid fit of ``CONTACT_LOCAL`` onto the observed
    points, which is what lets a noisy detection still yield a usable heading.
    """
    q = np.asarray(contacts, float).reshape(-1, 4, 2)
    centre = q.mean(axis=1)
    dev = q - centre[:, None, :]
    ref = CONTACT_LOCAL - CONTACT_LOCAL.mean(axis=0)
    # Optimal planar rotation from the cross/dot accumulations (Kabsch in 2-D).
    sin = (ref[:, 0] * dev[..., 1] - ref[:, 1] * dev[..., 0]).sum(axis=1)
    cos = (ref[:, 0] * dev[..., 0] + ref[:, 1] * dev[..., 1]).sum(axis=1)
    heading = np.arctan2(sin, cos)
    return centre, heading


def track_coordinates(p):
    """Arc length along the centreline and signed lateral offset (left positive).

    This is the frame a track-limit decision is actually made in: the margin
    depends only on the lateral coordinate, so an error that lives purely in the
    along-track coordinate does not change any verdict.
    """
    p = np.atleast_2d(np.asarray(p, float))
    x, y = p[:, 0], p[:, 1]

    s_entry = np.clip(x, ENTRY_S, 0.0)
    lat_entry = y
    d_entry = np.hypot(x - s_entry, y)

    a = np.clip(np.arctan2(x, RADIUS - y), 0.0, np.pi / 2)
    cx, cy = RADIUS * np.sin(a), RADIUS * (1 - np.cos(a))
    nx, ny = -np.sin(a), np.cos(a)
    s_arc = a * RADIUS
    lat_arc = (x - cx) * nx + (y - cy) * ny
    d_arc = np.hypot(x - cx, y - cy)

    s_exit = ARC_LEN + np.clip(y - RADIUS, 0.0, 30.0)
    lat_exit = -(x - RADIUS)
    d_exit = np.hypot(x - RADIUS, y - np.clip(y, RADIUS, RADIUS + 30.0))

    d = np.stack([d_entry, d_arc, d_exit])
    pick = d.argmin(axis=0)
    s = np.choose(pick, [s_entry, s_arc, s_exit])
    lat = np.choose(pick, [lat_entry, lat_arc, lat_exit])
    return s, lat
