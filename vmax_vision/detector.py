"""Inference wrappers.

``VmaxDetector`` runs the network trained on this footage and returns boxes,
scores, silhouette masks and the four tyre-contact points per car.
``CocoYoloDetector`` runs an off-the-shelf COCO YOLOv8-seg with no fine-tuning
at all, as the zero-shot baseline the original plan called for: it answers
"does a general-purpose car detector already work on this imagery?" and gives
the trained model something to be measured against.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import CONTACT_SCALE, STRIDE, VmaxNet


def crop_mask(mask, box):
    """The instance silhouette, clipped to the integer box it belongs to."""
    h, w = mask.shape
    x0 = int(max(np.floor(box[0]), 0))
    y0 = int(max(np.floor(box[1]), 0))
    x1 = int(min(np.ceil(box[2]) + 1, w))
    y1 = int(min(np.ceil(box[3]) + 1, h))
    if x1 <= x0 or y1 <= y0:
        return {"origin": [x0, y0], "data": np.zeros((0, 0), bool)}
    return {"origin": [x0, y0], "data": np.ascontiguousarray(mask[y0:y1, x0:x1])}


def _pad_to(image, multiple=64):
    h, w = image.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return image, h, w


class VmaxDetector:
    """VMAX-Net at inference: peaks on the centre heatmap, then per-peak geometry."""

    name = "vmaxnet"

    def __init__(self, weights="pipeline_out/vmaxnet.pt", threshold=0.25,
                 max_detections=8, threads=4, device="cpu"):
        torch.set_num_threads(threads)
        blob = torch.load(weights, map_location=device, weights_only=False)
        self.model = VmaxNet()
        self.model.load_state_dict(blob["model"])
        self.device = str(device)
        self.model.eval().to(device=device, memory_format=torch.channels_last)
        self.threshold = threshold
        self.max_detections = max_detections
        self.step = int(blob.get("step", 0))
        self.weights = str(pathlib.Path(weights))

    @torch.no_grad()
    def __call__(self, frame_bgr):
        padded, h, w = _pad_to(frame_bgr)
        x = torch.from_numpy(padded).permute(2, 0, 1)[None].float()
        x = ((x - 114.0) / 58.0).to(self.device).contiguous(memory_format=torch.channels_last)
        out = self.model(x)

        heat = torch.sigmoid(out["heat"])[0, 0]
        peak = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
        keep = (heat == peak) & (heat >= self.threshold)
        ys, xs = torch.nonzero(keep, as_tuple=True)
        if len(ys) > self.max_detections:
            top = torch.argsort(heat[ys, xs], descending=True)[: self.max_detections]
            ys, xs = ys[top], xs[top]

        mask_prob = torch.sigmoid(out["mask"])[0, 0].cpu().numpy()
        mask_full = cv2.resize(mask_prob, (padded.shape[1], padded.shape[0]),
                               interpolation=cv2.INTER_LINEAR)[:h, :w] > 0.5

        detections = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            score = float(heat[y, x])
            off = out["offset"][0, :, y, x].cpu().numpy()
            size = np.exp(np.clip(out["size"][0, :, y, x].cpu().numpy(), -4, 7)) * STRIDE
            cx = (x + float(off[0])) * STRIDE
            cy = (y + float(off[1])) * STRIDE
            contacts = out["contacts"][0, :, y, x].cpu().numpy().reshape(4, 2) * CONTACT_SCALE
            contacts = contacts + (cx, cy)
            box = np.array([cx - size[0] / 2, cy - size[1] / 2,
                            cx + size[0] / 2, cy + size[1] / 2])
            if box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
                continue
            detections.append({
                "score": score,
                "box": box.tolist(),
                "centre": [cx, cy],
                "contacts_uv": contacts.tolist(),
                # Only the silhouette inside the box is ever used (livery
                # sampling), and keeping it box-aligned makes a whole clip's
                # detections small enough to cache and re-judge under a
                # different calibration without re-running the network.
                "mask_crop": crop_mask(mask_full, box),
            })
        detections.sort(key=lambda d: -d["score"])
        return detections


class CocoYoloDetector:
    """Zero-shot COCO YOLOv8-seg, vehicle classes only. Boxes and masks, no contacts."""

    name = "yolov8-seg-coco"
    VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

    def __init__(self, weights="yolov8m-seg.pt", threshold=0.15, imgsz=1280):
        from ultralytics import YOLO  # imported lazily: the baseline is optional
        self.model = YOLO(weights)
        self.threshold = threshold
        self.imgsz = imgsz
        self.weights = weights

    def __call__(self, frame_bgr):
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.threshold,
                                 verbose=False)[0]
        detections = []
        if res.boxes is None:
            return detections
        masks = None
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()
        for i in range(len(res.boxes)):
            cls = int(res.boxes.cls[i])
            if cls not in self.VEHICLE_CLASSES:
                continue
            box = res.boxes.xyxy[i].cpu().numpy()
            mask_crop = None
            if masks is not None and i < len(masks):
                full = cv2.resize(masks[i], (frame_bgr.shape[1], frame_bgr.shape[0])) > 0.5
                mask_crop = crop_mask(full, box)
            detections.append({
                "score": float(res.boxes.conf[i]),
                "box": box.tolist(),
                "centre": [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
                "contacts_uv": None,
                "coco_class": self.VEHICLE_CLASSES[cls],
                "mask_crop": mask_crop,
            })
        detections.sort(key=lambda d: -d["score"])
        return detections
