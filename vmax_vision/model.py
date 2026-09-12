"""VMAX-Net: a compact anchor-free car detector with a tyre-contact head.

Boxes and silhouettes are not enough for a track-limit decision. Bodywork and
wings overhang the tyres, and the ground-truth contract is explicit that they
must never decide a verdict, so the network predicts the four *tyre contact
points* directly. Those lie on the z=0 plane by construction, which is the one
plane a single ground homography can invert exactly -- so contacts, unlike any
part of a bounding box, back-project to metres without a depth assumption.

Heads, all at stride 4:
  ``heat``     car centre likelihood
  ``offset``   sub-cell centre refinement
  ``size``     box width and height in pixels
  ``contacts`` the four contact points as offsets from the centre
  ``mask``     class-agnostic car segmentation
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
CONTACT_SCALE = 64.0  # contact offsets are regressed in units of 64 px


def conv_bn(cin, cout, stride=1, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = conv_bn(ch, ch)
        self.b = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.b(self.a(x)))


class VmaxNet(nn.Module):
    def __init__(self, width=(28, 56, 88, 128, 160), head=40):
        super().__init__()
        c1, c2, c3, c4, c5 = width
        self.stem = nn.Sequential(conv_bn(3, 20, 2), conv_bn(20, c1, 2))
        self.down2 = nn.Sequential(conv_bn(c1, c2, 2), Residual(c2), Residual(c2))
        self.down3 = nn.Sequential(conv_bn(c2, c3, 2), Residual(c3), Residual(c3))
        self.down4 = nn.Sequential(conv_bn(c3, c4, 2), Residual(c4))
        # A fifth stage exists for scale, not capacity: a car filling the frame
        # is wider than the receptive field of a four-stage stem, so its centre
        # cell cannot see the whole object and the heatmap never peaks there.
        self.down5 = nn.Sequential(conv_bn(c4, c5, 2), Residual(c5))
        self.lat5 = nn.Conv2d(c5, head, 1)
        self.lat3 = nn.Conv2d(c3, head, 1)
        self.lat2 = nn.Conv2d(c2, head, 1)
        self.lat1 = nn.Conv2d(c1, head, 1)
        self.lat4 = nn.Conv2d(c4, head, 1)
        self.smooth = nn.Sequential(conv_bn(head, head), Residual(head))
        self.heat = nn.Conv2d(head, 1, 1)
        self.offset = nn.Conv2d(head, 2, 1)
        self.size = nn.Conv2d(head, 2, 1)
        self.contacts = nn.Conv2d(head, 8, 1)
        self.mask = nn.Conv2d(head, 1, 1)
        nn.init.constant_(self.heat.bias, -4.0)   # rare positives: start pessimistic
        nn.init.constant_(self.mask.bias, -4.0)

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        f5 = self.down5(f4)
        p = self.lat5(f5)
        for lat, feat in ((self.lat4, f4), (self.lat3, f3), (self.lat2, f2), (self.lat1, f1)):
            p = F.interpolate(p, size=feat.shape[-2:], mode="nearest") + lat(feat)
        p = self.smooth(p)
        return {
            "heat": self.heat(p),
            "offset": self.offset(p),
            "size": self.size(p),
            "contacts": self.contacts(p),
            "mask": self.mask(p),
        }


def focal_loss(pred_logits, target, alpha=2.0, beta=4.0):
    """CornerNet/CenterNet penalty-reduced focal loss on a Gaussian target."""
    pred = torch.clamp(torch.sigmoid(pred_logits), 1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = -torch.log(1 - pred) * pred ** alpha * (1 - target) ** beta * neg
    n = pos.sum()
    if n == 0:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / n


def gather_at(feature, index):
    """Sample a (B,C,H,W) map at flat (B,N) cell indices -> (B,N,C)."""
    b, c, h, w = feature.shape
    flat = feature.view(b, c, h * w)
    idx = index.unsqueeze(1).expand(b, c, index.shape[1])
    return flat.gather(2, idx).permute(0, 2, 1)


def detection_loss(out, target, weights=(1.0, 3.0, 1.0, 2.0, 1.0)):
    w_heat, w_contact, w_size, w_offset, w_mask = weights
    valid = target["valid"]
    n = valid.sum().clamp(min=1.0)
    # Contacts are regressed in absolute pixels but scored relative to the car's
    # own size, so a frame-filling car and a distant one contribute comparably.
    # Absolute pixels stay the prediction, so contact accuracy never inherits
    # the size head's error.
    scale = target["contact_norm"].unsqueeze(-1)

    loss_heat = focal_loss(out["heat"], target["heat"])
    loss_mask = F.binary_cross_entropy_with_logits(out["mask"], target["mask"])

    def masked_l1(pred_map, key, weight=None):
        pred = gather_at(pred_map, target["index"])
        err = torch.abs(pred - target[key]) * valid.unsqueeze(-1)
        if weight is not None:
            err = err * weight
        return err.sum() / (n * pred.shape[-1])

    loss_contact = masked_l1(out["contacts"], "contacts", scale)
    loss_size = masked_l1(out["size"], "size")
    loss_offset = masked_l1(out["offset"], "offset")
    total = (w_heat * loss_heat + w_contact * loss_contact + w_size * loss_size
             + w_offset * loss_offset + w_mask * loss_mask)
    return total, {"heat": loss_heat.item(), "contact": loss_contact.item(),
                   "size": loss_size.item(), "offset": loss_offset.item(),
                   "mask": loss_mask.item()}
