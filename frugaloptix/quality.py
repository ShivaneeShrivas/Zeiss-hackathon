"""Image-quality gate + restoration-uncertainty gate (colleague points 15, 25, 27).
Two separate outputs so an unusable image is never mistaken for a 'normal' scan:
   quality  : ok | recapture(reason)   - blur, illumination, retinal coverage, noise (from the CAPTURED image)
   trust    : trusted | untrusted      - disagreement between restorer ensemble members / flipped views
Thresholds are fitted on VALIDATION images only (percentiles of the accepted reference distribution)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import laplace, gaussian_filter


def iqa(img_u8: np.ndarray) -> dict:
    g = img_u8[..., 1].astype(np.float32) / 255.0
    inside = g > 0.04
    cov = float(inside.mean())
    sharp = float(np.var(laplace(gaussian_filter(g, 0.7)))[()] if inside.any() else 0.0)
    noise = float(np.std(g - gaussian_filter(g, 1.0))[()])
    vals = g[inside] if inside.any() else g.ravel()
    return dict(coverage=cov, sharpness=sharp * 1e4, mean_level=float(vals.mean()), contrast=float(np.percentile(vals, 95) - np.percentile(vals, 5)), noise=noise)


def fit_thresholds(imgs_u8, q_low=2.0, q_high=98.0) -> dict:
    """Reference images = validation images captured with the REFERENCE (glass) optics."""
    rows = [iqa(i) for i in imgs_u8]
    col = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    return dict(min_coverage=float(np.percentile(col("coverage"), q_low) * 0.9), min_contrast=float(np.percentile(col("contrast"), q_low) * 0.9),
                level_range=[float(np.percentile(col("mean_level"), q_low) * 0.8), float(np.percentile(col("mean_level"), q_high) * 1.2)],
                max_noise=float(np.percentile(col("noise"), q_high) * 1.5), n_reference=len(rows))


def quality_check(img_u8, th: dict, min_sharp: float | None = None) -> dict:
    m = iqa(img_u8)
    why = []
    if m["coverage"] < th["min_coverage"]: why.append("retina not fully in view")
    if m["contrast"] < th["min_contrast"]: why.append("low contrast / out of focus")
    if not (th["level_range"][0] <= m["mean_level"] <= th["level_range"][1]): why.append("illumination outside range")
    if m["noise"] > th["max_noise"]: why.append("too noisy")
    if min_sharp is not None and m["sharpness"] < min_sharp: why.append("too blurred to restore reliably")
    return dict(ok=not why, reasons=why, metrics=m)


@torch.no_grad()
def restoration_uncertainty(nets, x: torch.Tensor) -> torch.Tensor:
    """Mean absolute disagreement over ensemble members x {original, horizontally flipped} views. Higher = less trustworthy."""
    outs = []
    for n in nets:
        outs.append(n(x))
        outs.append(n(x.flip(3)).flip(3))
    o = torch.stack(outs)
    return (o - o.mean(0, keepdim=True)).abs().mean(dim=(0, 2, 3, 4))
