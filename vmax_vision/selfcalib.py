"""Ground-plane self-calibration from the video alone.

The pipeline is not allowed to read ``calibration.json`` when it judges a clip;
it has to work out where the camera is from the picture. What it may use is the
circuit map -- the same survey a steward's system would be given -- which fixes
the lateral offset of every painted and surfaced edge from the centreline.

Method: take the temporal median of the clip (the empty circuit), classify the
surface into asphalt / line paint / kerb red / apron / gravel / grass by colour,
and keep only the *pairwise* class borders, each of which is unambiguous and
corresponds to exactly one known offset. Then fit a six-parameter look-at
camera by chamfer matching the projected circuit map onto those borders:
a coarse vectorised random search followed by Powell refinement.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.optimize import least_squares, minimize

from . import track_model as tm
from .calib import homography_from_pose, apply_h

# Surface colour prototypes in BGR, read off the circuit's construction.
# Matching runs in Lab with a shading-tolerant weighting so the same surface
# under shadow or distance haze still lands on its own class.
PROTOTYPES = {
    "asphalt": (72, 68, 65),
    "line": (245, 245, 245),
    "kerb_red": (42, 34, 208),
    "apron": (103, 118, 43),
    "gravel": (137, 167, 181),
    "grass": (57, 112, 77),
    "sky": (201, 181, 148),
}

# Each usable border, with the lateral offset from the centreline that it marks.
BORDERS = [
    ("asphalt", "line", 6.90, 1.0),    # the track limit itself
    ("kerb_red", "apron", 7.80, 0.8),  # outer edge of the kerb
    ("apron", "gravel", 9.60, 0.8),    # painted apron meets the run-off
]


def classify_surface(bgr):
    """Per-pixel surface class index, and a validity mask excluding sky/scenery."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2Lab).astype(np.float32)
    names = list(PROTOTYPES)
    protos = np.stack([
        cv2.cvtColor(np.uint8([[PROTOTYPES[n]]]), cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]
        for n in names
    ])
    # Lightness is heavily discounted: shadow and distance haze move L, not hue.
    weight = np.array([0.35, 1.0, 1.0], np.float32)
    d = ((lab[:, :, None, :] - protos[None, None]) * weight) ** 2
    dist = d.sum(-1)
    label = dist.argmin(-1).astype(np.int16)
    valid = dist.min(-1) < 900.0
    return label, valid, names


def class_borders(bgr):
    """Binary masks of each class-pair border, keyed by the pair."""
    label, valid, names = classify_surface(bgr)
    idx = {n: i for i, n in enumerate(names)}
    masks = {n: ((label == i) & valid).astype(np.uint8) for n, i in idx.items()}
    kernel = np.ones((3, 3), np.uint8)
    borders = {}
    for a, b, _off, _w in BORDERS:
        # Two iterations bridge the one-pixel antialiased seam between two
        # painted surfaces, which is otherwise labelled as neither.
        grown_a = cv2.dilate(masks[a], kernel, iterations=2)
        grown_b = cv2.dilate(masks[b], kernel, iterations=2)
        touch = (grown_a & grown_b).astype(np.uint8)
        # Both sides must have real area nearby, which drops speckle and the
        # scenery above the horizon that happens to share a surface colour.
        area = cv2.blur(masks[a].astype(np.float32), (9, 9)) * cv2.blur(masks[b].astype(np.float32), (9, 9))
        touch[area < 0.02] = 0
        borders[(a, b)] = _largest_components(touch)
    return borders, masks


def _largest_components(mask, min_pixels=60, keep=12):
    """Drop border speckle: catch fencing and grandstand detail produce short
    fragments that share a surface colour but are not painted circuit edges."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    sizes = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(sizes)[::-1][:keep]
    out = np.zeros_like(mask)
    for i in order:
        if sizes[i] >= min_pixels:
            out[labels == i + 1] = 1
    return out


def _distance_fields(borders, shape):
    fields, offsets, weights = [], [], []
    for a, b, off, w in BORDERS:
        mask = borders[(a, b)]
        if mask.sum() < 40:
            continue
        dt = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
        fields.append(dt)
        offsets.append(off)
        weights.append(w)
    if not fields:
        raise RuntimeError("no usable surface borders found in this clip")
    return np.stack(fields), np.array(offsets), np.array(weights)


def _model_points(offsets, n=120):
    """Circuit-map sample points for every border curve, both sides."""
    s = np.linspace(tm.ENTRY_S, tm.EXIT_S, n)
    c, nrm = tm.centreline(s)
    pts, cls = [], []
    for k, off in enumerate(offsets):
        for side in (-1.0, 1.0):
            pts.append(c + side * nrm * off)
            cls.append(np.full(n, k))
    return np.concatenate(pts), np.concatenate(cls)


def _pack(pos, target, fov):
    return np.array([*pos, *target, fov], float)


def _unpack(theta):
    return theta[:3], np.array([theta[3], theta[4], 0.0]), theta[5]


def _homographies(theta, width, height):
    """Vectorised look-at homographies for a stack of parameter vectors."""
    theta = np.atleast_2d(theta)
    pos = theta[:, :3]
    tgt = np.column_stack([theta[:, 3], theta[:, 4], np.zeros(len(theta))])
    fwd = tgt - pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    right = np.cross(fwd, np.array([0, 0, 1.0]))
    nrm = np.linalg.norm(right, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1e-9
    right /= nrm
    down = np.cross(fwd, right)
    focal = height / 2 / np.tan(np.deg2rad(theta[:, 5]) / 2)
    K = np.zeros((len(theta), 3, 3))
    K[:, 0, 0] = focal
    K[:, 1, 1] = focal
    K[:, 0, 2] = width / 2
    K[:, 1, 2] = height / 2
    K[:, 2, 2] = 1.0
    R = np.stack([right, down, fwd], axis=1)
    t = -np.einsum("nij,nj->ni", R, pos)
    M = np.stack([R[:, :, 0], R[:, :, 1], t], axis=2)
    return K @ M


def _project(theta, pts, width, height):
    """Project ground points for a batch of poses. Returns u, v and in-front flag."""
    H = _homographies(theta, width, height)
    xy1 = np.hstack([pts, np.ones((len(pts), 1))])
    proj = np.einsum("nij,pj->npi", H, xy1)
    w = proj[:, :, 2]
    front = w > 1e-6
    safe = np.where(front, w, 1.0)
    return proj[:, :, 0] / safe, proj[:, :, 1] / safe, front


def _forward_cost(theta, fields, weights, model_pts, model_cls, width, height,
                  cap=24.0, trim=1.0, min_seen=40):
    """Batched map->image term: the visible circuit map must lie on the paint.

    Invisibility is not penalised -- a fixed camera legitimately sees a fraction
    of the circuit -- and the score is a trimmed mean, so map points that are
    genuinely occluded (behind a barrier, past a crest) do not swamp the fit.
    A spread guard rejects the degenerate poses that squeeze the whole map into
    a handful of pixels, which is the one thing a forward-only term will accept.
    """
    theta = np.atleast_2d(theta)
    u, v, front = _project(theta, model_pts, width, height)
    inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ui = np.clip(u, 0, width - 1).astype(np.int32)
    vi = np.clip(v, 0, height - 1).astype(np.int32)
    d = np.minimum(fields[model_cls[None, :], vi, ui], cap) * weights[model_cls][None, :]
    d = np.where(inside, d, np.inf).astype(np.float32)
    d.sort(axis=1)
    seen = inside.sum(1)
    k = np.maximum((seen * trim).astype(np.int64), 1)
    csum = np.cumsum(np.where(np.isfinite(d), d, 0.0), axis=1)
    cost = csum[np.arange(len(theta)), k - 1] / k
    cost = np.where(seen >= min_seen, cost, cap)

    umin = np.where(inside, u, np.inf).min(1)
    umax = np.where(inside, u, -np.inf).max(1)
    vmin = np.where(inside, v, np.inf).min(1)
    vmax = np.where(inside, v, -np.inf).max(1)
    with np.errstate(over="ignore", invalid="ignore"):
        span = np.hypot(np.clip(umax - umin, 0, 1e6), np.clip(vmax - vmin, 0, 1e6))
    span = np.nan_to_num(span)
    return cost + cap * np.clip(0.45 - span / np.hypot(width, height), 0, None) * 4.0


class _Reverse:
    """Image->map term: every detected border pixel must be explained.

    Rasterises the projected circuit map at reduced resolution and measures how
    far each detected border pixel sits from the curve of its own class. A pose
    that puts no curve where the paint is pays the truncation cap, which is what
    rules out the degenerate fits a forward-only term accepts.
    """

    def __init__(self, borders, offsets, width, height, scale=2, max_points=2500, seed=0):
        self.scale = scale
        self.w = width // scale
        self.h = height // scale
        rng = np.random.default_rng(seed)
        self.points = []
        used = list(offsets)
        for a, b, off, _w in BORDERS:
            if off not in used:
                continue
            ys, xs = np.nonzero(borders[(a, b)])
            if len(xs) > max_points:
                pick = rng.choice(len(xs), max_points, replace=False)
                xs, ys = xs[pick], ys[pick]
            self.points.append((np.clip(xs // scale, 0, self.w - 1).astype(np.int32),
                                np.clip(ys // scale, 0, self.h - 1).astype(np.int32)))
        s = np.linspace(tm.ENTRY_S, tm.EXIT_S, 260)
        c, nrm = tm.centreline(s)
        self.curves = [[c + side * nrm * off for side in (-1.0, 1.0)] for off in used]

    def _curve_distance(self, k, theta, width, height):
        """Distance in pixels from every image location to class ``k``'s curve."""
        canvas = np.ones((self.h, self.w), np.uint8)
        for curve in self.curves[k]:
            u, v, front = _project(np.asarray(theta, float)[None], curve, width, height)
            xy = np.column_stack([u[0], v[0]]) / self.scale
            ok = front[0] & np.isfinite(xy).all(1) & (np.abs(xy) < 1e5).all(1)
            start = 0
            for i in range(len(xy) + 1):
                if i == len(xy) or not ok[i]:
                    run = xy[start:i]
                    if len(run) >= 2:
                        cv2.polylines(canvas, [run.astype(np.int32)], False, 0, 1)
                    start = i + 1
        return cv2.distanceTransform(canvas, cv2.DIST_L2, 3) * self.scale

    def __call__(self, theta, width, height, cap=12.0):
        total, count = 0.0, 0
        for k in range(len(self.curves)):
            dt = self._curve_distance(k, theta, width, height)
            xs, ys = self.points[k]
            total += float(np.minimum(dt[ys, xs], cap).sum())
            count += len(xs)
        return total / max(count, 1)

    def coverage(self, theta, width, height, tau=3.0):
        """Fraction of detected paint that the projected map actually explains.

        A hard threshold, unlike a mean distance, is dominated by whether whole
        stretches of paint are covered at all -- which is precisely the question
        that separates the true rotation of a corner from a plausible-looking
        wrong one, because only the straights fall outside a wrong rotation.
        """
        hit = total = 0
        for k in range(len(self.curves)):
            dt = self._curve_distance(k, theta, width, height)
            xs, ys = self.points[k]
            hit += int((dt[ys, xs] < tau).sum())
            total += len(xs)
        return hit / max(total, 1)


BOUNDS_LO = np.array([-45.0, -55.0, 1.5, -16.0, -16.0, 18.0])
BOUNDS_HI = np.array([95.0, 80.0, 30.0, 60.0, 64.0, 72.0])


def _sweep(cost_fn, rng, count, batch=12000):
    """Uniform random sweep over the pose bounds, returning scored candidates."""
    thetas, costs = [], []
    done = 0
    while done < count:
        n = min(batch, count - done)
        done += n
        theta = BOUNDS_LO + rng.random((n, 6)) * (BOUNDS_HI - BOUNDS_LO)
        # A fixed camera stands off the circuit and points at it: both facts
        # cut the search volume by roughly an order of magnitude.
        theta = theta[(tm.distance_to_centreline(theta[:, :2]) > 10.0)
                      & (tm.distance_to_centreline(theta[:, 3:5]) < 14.0)]
        if not len(theta):
            continue
        thetas.append(theta)
        costs.append(cost_fn(theta))
    return np.concatenate(thetas), np.concatenate(costs)


def _diverse(pool, cost, count, min_gap=0.04):
    """Greedy diversity selection: keep the best candidate of each basin.

    A population that has collapsed onto one basin cannot be rescued by a local
    optimiser, so the finalists are chosen for spread as well as score.
    """
    span = BOUNDS_HI - BOUNDS_LO
    picked = []
    for i in np.argsort(cost):
        cand = pool[i]
        if all(np.linalg.norm((cand - pool[j]) / span) > min_gap for j in picked):
            picked.append(i)
            if len(picked) >= count:
                break
    return pool[picked]


def _evolve(cost_fn, parents, rng, generations=12, children=6, keep=1200,
            sigma0=0.13, decay=0.72):
    """Shrinking-neighbourhood population search: the coarse sweep is far too
    sparse to land in a six-dimensional basin on its own."""
    span = BOUNDS_HI - BOUNDS_LO
    pool = parents
    pool_cost = cost_fn(pool)
    for gen in range(generations):
        sigma = sigma0 * (decay ** gen) * span
        kids = (np.repeat(pool, children, axis=0)
                + rng.normal(size=(len(pool) * children, 6)) * sigma)
        kids = np.clip(kids, BOUNDS_LO, BOUNDS_HI)
        allc = np.concatenate([pool, kids])
        cost = np.concatenate([pool_cost, cost_fn(kids)])
        order = np.argsort(cost)[:keep]
        pool, pool_cost = allc[order], cost[order]
    return pool, pool_cost


ARC_CENTRE = np.array([0.0, tm.RADIUS])


def _gauge_variants(theta, count=241, span_deg=92.0):
    """Poses related to ``theta`` by a rotation about the corner's centre.

    A camera that mostly sees the constant-radius part of a corner is almost
    unconstrained along the arc: rotating the whole rig about the centre of the
    circle reproduces nearly the same picture. The straights break the tie, but
    only weakly, so the ambiguity is enumerated explicitly rather than left for
    a local optimiser to stumble across.
    """
    theta = np.asarray(theta, float)
    angles = np.deg2rad(np.linspace(-span_deg, span_deg, count))
    cos, sin = np.cos(angles), np.sin(angles)
    out = np.repeat(theta[None], count, axis=0)
    for a, b in ((0, 1), (3, 4)):
        dx = theta[a] - ARC_CENTRE[0]
        dy = theta[b] - ARC_CENTRE[1]
        out[:, a] = ARC_CENTRE[0] + cos * dx - sin * dy
        out[:, b] = ARC_CENTRE[1] + sin * dx + cos * dy
    return out


def _gauge_orthogonal_basis(theta):
    """Orthonormal basis of the five pose directions that are not the corner's
    rotational gauge, evaluated at ``theta``."""
    theta = np.asarray(theta, float)
    g = np.zeros(6)
    g[0], g[1] = -(theta[1] - ARC_CENTRE[1]), theta[0] - ARC_CENTRE[0]
    g[3], g[4] = -(theta[4] - ARC_CENTRE[1]), theta[3] - ARC_CENTRE[0]
    norm = np.linalg.norm(g)
    if norm < 1e-9:
        return np.eye(6)
    g /= norm
    q, _ = np.linalg.qr(np.column_stack([g, np.eye(6)]))
    return q[:, 1:6]


class _GroundResidual:
    """Final polish in the units the answer is given in.

    Every detected border pixel is back-projected to the ground and asked for
    its lateral offset from the centreline; the residual is how far that offset
    is from the surveyed one. Minimising this directly minimises the quantity a
    margin judgement inherits, rather than a pixel distance that means
    centimetres near the camera and metres at the horizon.
    """

    def __init__(self, borders, offsets, width, height, max_points=1600, seed=0):
        self.width, self.height = width, height
        rng = np.random.default_rng(seed)
        pix, nominal, weight = [], [], []
        used = list(offsets)
        for a, b, off, w in BORDERS:
            if off not in used:
                continue
            ys, xs = np.nonzero(borders[(a, b)])
            if not len(xs):
                continue
            if len(xs) > max_points:
                pick = rng.choice(len(xs), max_points, replace=False)
                xs, ys = xs[pick], ys[pick]
            pix.append(np.column_stack([xs + 0.5, ys + 0.5]).astype(float))
            nominal.append(np.full(len(xs), off))
            weight.append(np.full(len(xs), w))
        self.pix = np.concatenate(pix)
        self.nominal = np.concatenate(nominal)
        self.weight = np.concatenate(weight)

        self.active = np.ones(len(self.pix), bool)

    def offsets_at(self, theta):
        pos, target, fov = _unpack(np.asarray(theta, float))
        H = homography_from_pose(pos, target, fov, self.width, self.height)
        ground = apply_h(np.linalg.inv(H), self.pix)
        return tm.distance_to_centreline(ground)

    def __call__(self, theta):
        try:
            r = (self.offsets_at(theta) - self.nominal) * self.weight
        except np.linalg.LinAlgError:
            return np.full(int(self.active.sum()), 1e3)
        r = np.clip(np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3), -50, 50)
        return r[self.active]

    def fit(self, theta, rounds=4):
        """Iteratively reweighted fit in the gauge-orthogonal subspace.

        Lateral offset is exactly invariant to rotating the rig about the
        corner's centre, so an unconstrained metric fit would happily slide
        along that direction and undo the gauge the chamfer stage resolved from
        the straights. The polish is therefore run in the five directions
        orthogonal to it and only corrects what it can actually see.
        """
        theta = np.asarray(theta, float)
        for _ in range(rounds):
            basis = _gauge_orthogonal_basis(theta)
            base = theta.copy()

            def wrapped(u):
                return self(base + basis @ u)

            out = least_squares(wrapped, np.zeros(basis.shape[1]), loss="soft_l1",
                                f_scale=0.15, xtol=1e-13, ftol=1e-13, max_nfev=600)
            theta = np.clip(base + basis @ np.asarray(out.x, float), BOUNDS_LO, BOUNDS_HI)
            try:
                r = np.abs(self.offsets_at(theta) - self.nominal)
            except np.linalg.LinAlgError:
                break
            r = np.nan_to_num(r, nan=1e3, posinf=1e3)
            scale = max(1.4826 * float(np.median(r[np.isfinite(r)])), 0.05)
            keep = r < max(4.0 * scale, 0.60)
            if keep.sum() < 200 or (keep == self.active).all():
                self.active = keep if keep.sum() >= 200 else self.active
                break
            self.active = keep
        return theta

    def rms(self, theta):
        r = self(theta)
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")


def paint_agreement(background_bgr, position, look_at, fov, seed=0):
    """Score any camera solution by the same measure the fit maximises.

    Used to tell a search failure from an objective failure: if the true camera
    scores *worse* than the recovered one, no amount of searching would have
    found it, and the ambiguity is in what the picture can show.
    """
    height, width = background_bgr.shape[:2]
    borders, _masks = class_borders(background_bgr)
    fields, offsets, _weights = _distance_fields(borders, background_bgr.shape[:2])
    reverse = _Reverse(borders, offsets, width, height, seed=seed)
    pts, cls = _model_points(offsets, n=340)
    theta = _pack(np.asarray(position, float), np.asarray(look_at, float)[:2], float(fov))
    u, v, front = _project(theta[None], pts, width, height)
    inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ui = np.clip(u, 0, width - 1).astype(np.int32)
    vi = np.clip(v, 0, height - 1).astype(np.int32)
    near = fields[cls[None, :], vi, ui] < 3.0
    model_cov = float((near & inside).sum() / max(inside.sum(), 1))
    return (model_cov + reverse.coverage(theta, width, height)) / 2.0


def estimate_homography(background_bgr, seed=0, sweep=2000000, keep=2000,
                        finalists=60, refine=6, verbose=False):
    """Fit the ground homography of a fixed camera from one background image.

    Sweep the pose space, evolve the survivors under the cheap forward term,
    rescore the finalists with the full symmetric cost, then polish with a
    derivative-free local search. Returns the homography plus the recovered
    camera pose, which is a by-product a steward can sanity-check on a map.
    """
    height, width = background_bgr.shape[:2]
    borders, _masks = class_borders(background_bgr)
    fields, offsets, weights = _distance_fields(borders, background_bgr.shape[:2])
    reverse = _Reverse(borders, offsets, width, height, seed=seed)

    coarse_pts, coarse_cls = _model_points(offsets, n=60)
    mid_pts, mid_cls = _model_points(offsets, n=150)
    dense_pts, dense_cls = _model_points(offsets, n=340)

    def forward(pts, cls, trim=0.72, batch=8000):
        # The search runs on a trimmed score, which tolerates map points that
        # are occluded and so keeps a sharp basin; scoring and gauge selection
        # run untrimmed, because the parts of the map that a wrong rotation
        # fails to explain are exactly the parts a trimmed score throws away.
        def fn(theta):
            out = []
            for i in range(0, len(theta), batch):
                out.append(_forward_cost(theta[i:i + batch], fields, weights, pts, cls,
                                         width, height, trim=trim))
            return np.concatenate(out) if out else np.zeros(0)
        return fn

    def coverage(theta):
        """Two-sided hard-threshold agreement, used to choose among rotations."""
        theta = np.asarray(theta, float)
        if np.any(theta < BOUNDS_LO) or np.any(theta > BOUNDS_HI):
            return -1.0
        u, v, front = _project(theta[None], dense_pts, width, height)
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ui = np.clip(u, 0, width - 1).astype(np.int32)
        vi = np.clip(v, 0, height - 1).astype(np.int32)
        near = fields[dense_cls[None, :], vi, ui] < 3.0
        model_cov = float((near & inside).sum() / max(inside.sum(), 1))
        return model_cov + reverse.coverage(theta, width, height)

    def best_gauge(theta, count=241, span_deg=92.0):
        variants = _gauge_variants(theta, count=count, span_deg=span_deg)
        scores = [coverage(v) for v in variants]
        return variants[int(np.argmax(scores))], float(np.max(scores))

    def symmetric(theta):
        theta = np.asarray(theta, float)
        excess = np.abs(np.clip(theta, BOUNDS_LO, BOUNDS_HI) - theta).sum()
        if excess > 0:
            return 1e4 + float(excess)
        fwd = float(_forward_cost(theta[None], fields, weights, dense_pts, dense_cls,
                                  width, height, trim=1.0)[0])
        return fwd + 3.0 * reverse(theta, width, height)

    rng = np.random.default_rng(seed)
    theta, cost = _sweep(forward(coarse_pts, coarse_cls), rng, sweep)
    order = np.argsort(cost)[:keep]
    if verbose:
        print(f"  sweep best forward {cost[order[0]]:.3f}")
    coarse_fn = forward(coarse_pts, coarse_cls)  # trimmed, for the sweep
    seeds = [theta[order]]
    for t in theta[order[:400]]:
        seeds.append(_gauge_variants(t, count=65))
    seeds = np.concatenate(seeds)
    seed_cost = coarse_fn(seeds)
    seeds = seeds[np.argsort(seed_cost)[:keep]]
    pool, pool_cost = _evolve(forward(mid_pts, mid_cls), seeds, rng, keep=keep)
    if verbose:
        print(f"  evolved best forward {pool_cost[0]:.3f}")

    # Enumerate the corner's rotational gauge around each survivor before any
    # local polish, then keep the distinct basins.
    # Align each distinct basin to its own best rotation before comparing them:
    # an otherwise-correct pose at the wrong rotation must not be discarded.
    picks = _diverse(pool, pool_cost, finalists)
    aligned = []
    for t in picks:
        gt, gcov = best_gauge(t, count=121)
        aligned.append((gcov, gt))
    aligned.sort(key=lambda kv: -kv[0])
    if verbose:
        print(f"  best aligned coverage {aligned[0][0] / 2:.3f}")
    shortlist = sorted(((symmetric(t), t) for _c, t in aligned[:refine * 2]), key=lambda kv: kv[0])
    results = []
    for cost0, theta0 in shortlist[:refine]:
        res = minimize(symmetric, theta0, method="Nelder-Mead",
                       options=dict(maxiter=1400, xatol=1e-4, fatol=1e-6))
        best = (float(res.fun), np.asarray(res.x, float)) if res.fun < cost0 else (cost0, theta0)
        # One more pass through the gauge, chosen by coverage rather than by the
        # mean cost, then polished locally.
        gauge_theta, _ = best_gauge(best[1])
        polish = minimize(symmetric, gauge_theta, method="Nelder-Mead",
                          options=dict(maxiter=1400, xatol=1e-4, fatol=1e-6))
        gauged = (float(polish.fun), np.asarray(polish.x, float))
        results.append(max([best, gauged], key=lambda kv: coverage(kv[1])))
        if verbose:
            print(f"  {cost0:8.3f} -> {best[0]:8.3f}  {np.round(best[1], 2)}")
    results.sort(key=lambda kv: -coverage(kv[1]))
    cost, theta = results[0]

    residual = _GroundResidual(borders, offsets, width, height, seed=seed)
    polished = residual.fit(theta)
    if coverage(polished) >= coverage(theta) - 0.01:
        theta, cost = polished, symmetric(polished)
    rms = residual.rms(theta)
    cover = coverage(theta)
    if verbose:
        print(f"  ground polish rms offset residual {rms:.4f} m")
    pos, target, fov = _unpack(theta)
    H = homography_from_pose(pos, target, fov, width, height)
    return {
        "homography": H,
        "cost": float(cost),
        "position_m": [float(x) for x in pos],
        "look_at_m": [float(x) for x in target],
        "vertical_fov_deg": float(fov),
        "border_pixels": {f"{a}|{b}": int(borders[(a, b)].sum()) for a, b, _o, _w in BORDERS},
        "resolution": [width, height],
        "offset_residual_rms_m": rms,
        "paint_agreement": float(cover / 2.0),
        "method": "surface-border chamfer against the circuit map; video only",
    }
