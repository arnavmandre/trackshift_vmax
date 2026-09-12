"""Crop sampling and target construction for VMAX-Net."""
from __future__ import annotations

import cv2
import numpy as np

from . import dataset as ds
from .model import CONTACT_SCALE, STRIDE

MAX_INSTANCES = 4


def gaussian_radius(h, w, min_overlap=0.5):
    """CenterNet's radius: how far a centre may drift and still overlap enough."""
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / 2
    return max(min(r1, r2, r3), 1.0)


def draw_gaussian(heat, cx, cy, radius):
    sigma = max(radius / 3.0, 0.6)
    size = int(radius) * 2 + 1
    ax = np.arange(-int(radius), int(radius) + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma ** 2))
    h, w = heat.shape
    x0, x1 = max(0, cx - int(radius)), min(w, cx + int(radius) + 1)
    y0, y1 = max(0, cy - int(radius)), min(h, cy + int(radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    gx0, gy0 = x0 - (cx - int(radius)), y0 - (cy - int(radius))
    patch = g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
    np.maximum(heat[y0:y1, x0:x1], patch, out=heat[y0:y1, x0:x1])


class CropSampler:
    """Random scaled crops around the cars, plus a share of empty background."""

    def __init__(self, samples, crop=384, seed=0, scale=(0.5, 4.5),
                 background_share=0.15, perspective=0.11):
        self.samples = samples
        self.crop = crop
        self.scale = scale
        self.background_share = background_share
        self.perspective = perspective
        self.rng = np.random.default_rng(seed)
        self.jpeg = []
        for s in samples:
            ok, buf = cv2.imencode(".jpg", s["image"], [cv2.IMWRITE_JPEG_QUALITY, 94])
            self.jpeg.append(buf if ok else None)

    def _image(self, i):
        return cv2.imdecode(self.jpeg[i], cv2.IMREAD_COLOR)

    def _recolour_livery(self, img, mask, probability=0.65):
        """Rotate the hue of the car's own pixels, leaving the circuit alone."""
        if self.rng.random() > probability or not mask.any():
            return img
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        shift = int(self.rng.integers(15, 165))
        sel = mask.astype(bool)
        hue = hsv[:, :, 0].astype(np.int32)
        hue[sel] = (hue[sel] + shift) % 180
        hsv[:, :, 0] = hue.astype(np.uint8)
        if self.rng.random() < 0.5:
            sat = hsv[:, :, 1].astype(np.float32)
            sat[sel] *= float(self.rng.uniform(0.55, 1.35))
            hsv[:, :, 1] = np.clip(sat, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    def _photometric(self, img):
        img = img.astype(np.float32)
        img *= self.rng.uniform(0.75, 1.28)
        img += self.rng.uniform(-22, 22)
        if self.rng.random() < 0.35:
            img += self.rng.normal(0, self.rng.uniform(2, 7), img.shape)
        if self.rng.random() < 0.20:
            k = int(self.rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        return np.clip(img, 0, 255)

    def sample(self):
        i = int(self.rng.integers(len(self.samples)))
        rec = self.samples[i]
        img = self._image(i)
        h, w = img.shape[:2]
        # Log-uniform: the held-out viewpoint puts cars an order of magnitude
        # larger than the training cameras do, so scale has to be sampled across
        # decades rather than linearly around 1.
        scale = float(np.exp(self.rng.uniform(np.log(self.scale[0]), np.log(self.scale[1]))))
        crop = self.crop
        want = crop / scale

        if self.rng.random() < self.background_share:
            cx = self.rng.uniform(0, w)
            cy = self.rng.uniform(0, h)
        else:
            inst = rec["instances"][int(self.rng.integers(len(rec["instances"])))]
            x0, y0, x1, y1 = inst["box"]
            cx = (x0 + x1) / 2 + self.rng.uniform(-0.32, 0.32) * want
            cy = (y0 + y1) / 2 + self.rng.uniform(-0.32, 0.32) * want

        # A projective, not merely affine, crop. Two fixed viewpoints plus one
        # moving rig is thin coverage of "camera position"; warping each crop by
        # a random homography is the one augmentation that actually manufactures
        # new viewpoints, and it is exact for the ground-plane contact points
        # because a homography is what relates two views of a plane.
        sx0, sy0 = cx - want / 2, cy - want / 2
        corners = np.array([[sx0, sy0], [sx0 + want, sy0],
                            [sx0 + want, sy0 + want], [sx0, sy0 + want]], np.float32)
        jitter = self.rng.uniform(-self.perspective, self.perspective, (4, 2)) * want
        src = (corners + jitter).astype(np.float32)
        dst = np.array([[0, 0], [crop, 0], [crop, crop], [0, crop]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        patch = cv2.warpPerspective(img, M, (crop, crop), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        def to_crop(pts):
            pts = np.asarray(pts, float).reshape(1, -1, 2).astype(np.float32)
            return cv2.perspectiveTransform(pts, M)[0].astype(float)

        out = crop // STRIDE
        heat = np.zeros((out, out), np.float32)
        mask = np.zeros((crop, crop), np.uint8)
        index = np.zeros(MAX_INSTANCES, np.int64)
        valid = np.zeros(MAX_INSTANCES, np.float32)
        size = np.zeros((MAX_INSTANCES, 2), np.float32)
        offset = np.zeros((MAX_INSTANCES, 2), np.float32)
        contacts = np.zeros((MAX_INSTANCES, 8), np.float32)
        contact_norm = np.zeros(MAX_INSTANCES, np.float32)

        n = 0
        hulls = []
        for inst in rec["instances"]:
            hull = to_crop(inst["hull_uv"])
            hulls.append(hull)
            cv2.fillConvexPoly(mask, cv2.convexHull(hull.astype(np.float32)).astype(np.int32), 1)
            # The box is re-derived from the warped silhouette: transforming the
            # two original corners is only correct for an axis-aligned map.
            bx0, by0 = hull.min(axis=0)
            bx1, by1 = hull.max(axis=0)
            bw, bh = bx1 - bx0, by1 - by0
            ccx, ccy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if not (0 <= ccx < crop and 0 <= ccy < crop) or bw < 5 or bh < 5:
                continue
            if n >= MAX_INSTANCES:
                break
            fx, fy = ccx / STRIDE, ccy / STRIDE
            ix, iy = int(fx), int(fy)
            draw_gaussian(heat, ix, iy, gaussian_radius(bh / STRIDE, bw / STRIDE))
            heat[iy, ix] = 1.0
            index[n] = iy * out + ix
            valid[n] = 1.0
            # Log size: across a decade of object scale a raw-pixel target
            # would swamp every other head's gradient.
            size[n] = (np.log(bw / STRIDE), np.log(bh / STRIDE))
            offset[n] = (fx - ix, fy - iy)
            contacts[n] = ((to_crop(inst["contacts_uv"]) - (ccx, ccy)) / CONTACT_SCALE).reshape(-1)
            contact_norm[n] = CONTACT_SCALE / max(float(np.hypot(bw, bh)), 8.0)
            n += 1

        # Every training clip carries the same red car; the second entry only
        # ever appears in the withheld two-car scenarios. Rotating the livery
        # hue inside the silhouette is what stops the detector learning "car"
        # to mean "that shade of red".
        patch = self._recolour_livery(patch, mask)
        patch = self._photometric(patch)

        mask_small = cv2.resize(mask, (out, out), interpolation=cv2.INTER_AREA).astype(np.float32)
        return {
            "image": patch,
            "heat": heat[None],
            "mask": mask_small[None],
            "index": index,
            "valid": valid,
            "size": size,
            "offset": offset,
            "contacts": contacts,
            "contact_norm": contact_norm,
        }

    def batch(self, n):
        items = [self.sample() for _ in range(n)]
        return {k: np.stack([it[k] for it in items]) for k in items[0]}
