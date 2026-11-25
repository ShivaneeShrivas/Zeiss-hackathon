"""Synthetic fundus-like test image (disc, vessels, small bright lesions).

ONLY for exercising the pipeline before RFMiD/IDRiD are mounted. Never report it as a result.
"""
from __future__ import annotations

import numpy as np


def fundus_like(size: int = 256, seed: int = 0, n_lesions: int = 12, return_mask: bool = False, style: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    rr = np.hypot(yy - c, xx - c) / (size / 2)
    img = 0.55 - 0.25 * rr ** 2
    img[rr > 0.97] = 0.0
    for _ in range(14):  # vessels: random sinusoidal curves radiating from the optic disc
        ang = rng.uniform(0, 2 * np.pi)
        t = np.linspace(0.05, 0.9, 400)
        wob = 0.08 * np.sin(rng.uniform(3, 9) * t + rng.uniform(0, 6))
        x = c + (t * np.cos(ang + wob)) * size / 2
        y = c + (t * np.sin(ang + wob)) * size / 2
        for xi, yi in zip(x.astype(int), y.astype(int)):
            if 1 <= xi < size - 1 and 1 <= yi < size - 1:
                img[yi - 1:yi + 1, xi - 1:xi + 1] -= 0.18
    mask = np.zeros((size, size), bool)
    for _ in range(n_lesions):  # small bright exudate-like dots, biased toward the periphery
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.85)
        cx, cy = c + r * c * np.cos(a), c + r * c * np.sin(a)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6 ** 2))
        img += 0.30 * g
        mask |= g > 0.3
    img = np.clip(img, 0, 1)
    return (img, mask) if return_mask else img
