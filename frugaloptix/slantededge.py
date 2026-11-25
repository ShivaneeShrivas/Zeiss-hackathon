"""Slanted-edge MTF (ISO 12233 style) for the model-eye / bench measurements (colleague points 19, 23).
Validated here against an analytic Gaussian edge; point it at real captures with scripts/12_slanted_edge_mtf.py."""
from __future__ import annotations

import numpy as np


def esf_lsf(img: np.ndarray, oversample: int = 4):
    g = img.astype(float)
    g = g.mean(-1) if g.ndim == 3 else g
    h, w = g.shape
    # edge angle from the per-row 50% crossing positions
    pos = []
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    for r in range(h):
        row = (g[r] - lo) / max(hi - lo, 1e-9)
        k = np.where((row[:-1] < .5) & (row[1:] >= .5) | (row[:-1] >= .5) & (row[1:] < .5))[0]
        if len(k):
            k = k[0]
            pos.append(k + (0.5 - row[k]) / (row[k + 1] - row[k] + 1e-12))
        else:
            pos.append(np.nan)
    pos = np.array(pos)
    rows = np.arange(h)
    ok = np.isfinite(pos)
    slope, icpt = np.polyfit(rows[ok], pos[ok], 1)
    # project every pixel onto the edge normal -> super-resolved ESF
    xx = np.arange(w)[None, :] - (slope * rows[:, None] + icpt)
    bins = np.round(xx * oversample).astype(int)
    bins -= bins.min()
    esf = np.bincount(bins.ravel(), weights=g.ravel()) / np.maximum(np.bincount(bins.ravel()), 1)
    esf = esf[np.bincount(bins.ravel()) > 0]
    lsf = np.gradient(esf) * np.hanning(len(esf))
    return esf, lsf, oversample, float(np.degrees(np.arctan(slope)))


def mtf(img: np.ndarray, px_um: float, freqs_lp_mm=(10, 25, 50), oversample: int = 4) -> dict:
    esf, lsf, ov, ang = esf_lsf(img, oversample)
    F = np.abs(np.fft.rfft(lsf, n=8192))
    F /= F[0]
    f = np.fft.rfftfreq(8192, d=px_um / 1000.0 / ov)               # cycles/mm
    i50 = np.where(F < 0.5)[0]
    return dict(edge_angle_deg=ang, mtf_at={str(q): float(np.interp(q, f, F)) for q in freqs_lp_mm}, mtf50_lp_mm=float(f[i50[0]]) if len(i50) else None)
