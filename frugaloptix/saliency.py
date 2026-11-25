"""Stage 1 (Needs-to-Spec): Grad-CAM on the frozen classifier -> which image zones the disease task actually uses."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from frugaloptix.ml import device, to_tensor


def gradcam(clf, arr_u8, idx, bs: int = 16, dev=None) -> np.ndarray:
    """Mean Grad-CAM map (H,W) in [0,1] over images idx, w.r.t. the 'abnormal' logit of resnet18 layer4."""
    dev = dev or device()
    clf.eval().to(dev)
    acts, grads = {}, {}
    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))   # tensor hook: safe with in-place ReLU
    h1 = clf.net.layer4.register_forward_hook(fwd)
    h2 = h1
    total, n = None, 0
    for i in range(0, len(idx), bs):
        x = to_tensor(arr_u8[idx[i:i + bs]]).to(dev).requires_grad_(True)
        clf.zero_grad()
        clf(x).sum().backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = (w * acts["a"]).sum(1, keepdim=True)   # signed evidence; ReLU would zero everything when the logit is far below 0
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
        lo, hi = cam.amin(dim=(1, 2), keepdim=True), cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - lo) / (hi - lo + 1e-9)          # per-image min-max: relative evidence for "abnormal"
        s = cam.detach().sum(0).cpu().numpy()
        total = s if total is None else total + s
        n += len(cam)
    h1.remove()
    m = total / max(n, 1)
    return (m - m.min()) / (m.max() - m.min() + 1e-9)


def field_profile(cam: np.ndarray) -> list[float]:
    """Importance at field 0 / 0.5 / 1 (annuli [0,.25], [.25,.75], [.75,1] of the image radius), per unit area."""
    h, w = cam.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (min(h, w) / 2)
    return [float(cam[(rr >= lo) & (rr < hi)].mean()) for lo, hi in ((0, .25), (.25, .75), (.75, 1.0))]


def field_weights(profile: list[float], floor: float = 0.5, cap: float = 2.0) -> list[float]:
    if not np.all(np.isfinite(profile)) or np.mean(profile) <= 1e-9:
        return [1.0] * len(profile)   # degenerate saliency -> uniform weights (never NaN)
    p = np.asarray(profile) / np.mean(profile)
    return [float(v) for v in np.clip(p, floor, cap)]
