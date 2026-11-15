"""Field-dependent blur rendering of an image through a lens (v0).

APPROXIMATION: each field zone is blurred with a Gaussian whose sigma is set by the lens's
simulated RMS spot radius at that field (geometric spot, from Optiland), blended radially.
This captures "edge of field is worse" but not diffraction or coma shape. The planned
upgrade is Optiland's patch-wise wave PSF. Results from this module are SIMULATED.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def render(img: np.ndarray, rms_um_by_field: list[float], px_um: float = 10.0,
           noise_sigma: float = 0.0, seed: int = 0) -> np.ndarray:
    """img: HxW float in [0,1]. rms_um_by_field: RMS spot radius at normalised field 0, .5, 1."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (min(h, w) / 2)  # 0 centre .. 1 edge
    sig = [max(s / px_um, 1e-3) for s in rms_um_by_field]
    fields = np.array([0.0, 0.5, 1.0])
    stack = []
    for s in sig:
        if img.ndim == 3:
            stack.append(np.stack([gaussian_filter(img[..., c], s) for c in range(img.shape[2])], -1))
        else:
            stack.append(gaussian_filter(img, s))
    out = np.zeros_like(img, dtype=float)
    wts = [np.clip(1 - np.abs(rr - f) / 0.5, 0, 1) for f in fields]
    wsum = np.sum(wts, axis=0) + 1e-9
    for wt, layer in zip(wts, stack):
        wn = (wt / wsum)[..., None] if img.ndim == 3 else wt / wsum
        out += wn * layer
    if noise_sigma > 0:
        out = out + np.random.default_rng(seed).normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 1)
