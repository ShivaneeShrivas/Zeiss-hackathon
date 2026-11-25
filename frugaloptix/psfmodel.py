"""Field-dependent, wavelength-dependent PSFs from the real ray trace (Optiland), their MTF, and a renderer that uses them.
Replaces the single-Gaussian blur. Captures coma/astigmatism/spherical/lateral+axial colour as the ray trace gives them.
SIMULATED. Pixel pitch `px_um` is the *effective* sampling of the rendered image (SPEC['render_px_um'])."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate
from scipy.signal import fftconvolve

from frugaloptix import optics as op

CHANNEL_WL = {"R": 0.6563, "G": 0.5876, "B": 0.4861}   # C, d, F lines
N_FIELD = 5                                             # radial samples 0, .25, .5, .75, 1.0
KSIZE = 41


def _hist_kernel(x_um, y_um, px_um, ksize, centre):
    h = ksize // 2
    xe = (np.asarray(x_um) - centre[0]) / px_um + h + 0.5
    ye = (np.asarray(y_um) - centre[1]) / px_um + h + 0.5
    k = np.zeros((ksize, ksize))
    # bilinear splatting so sub-pixel structure and lateral-colour shifts survive
    x0, y0 = np.floor(xe).astype(int), np.floor(ye).astype(int)
    fx, fy = xe - x0, ye - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        ix, iy = x0 + dx, y0 + dy
        ok = (ix >= 0) & (ix < ksize) & (iy >= 0) & (iy < ksize)
        np.add.at(k, (iy[ok], ix[ok]), w[ok])
    return k


def _diffraction_sigma_px(wl_um, f_number, px_um):
    return max(0.44 * wl_um * f_number / px_um, 0.15)       # Airy core ~ Gaussian sigma 0.44*lambda*N


def psf_bank(design: dict, spec: dict, px_um: float, n_rings: int = 20, ksize: int = KSIZE):
    """-> {"kernels": array [N_FIELD, 3(R,G,B), k, k], "fields": fractions, "spot_rms_um": [N_FIELD,3]}.
    Geometric PSF of the real trace (field along +y), centred on the green chief-centroid (so colour shifts remain)."""
    lens = op.build_lens(design["elements"], spec["epd_mm"], spec["half_fov_deg"], design["back_mm"], fields=(0.0, 1.0))
    fn = spec["efl_mm"] / spec["epd_mm"]
    fields = np.linspace(0, 1, N_FIELD)
    ker = np.zeros((N_FIELD, 3, ksize, ksize))
    spot = np.zeros((N_FIELD, 3))
    def trace_um(f, wl):
        r = lens.trace(Hx=0.0, Hy=float(f), wavelength=wl, num_rays=n_rings, distribution="hexapolar")
        x, y = np.asarray(r.x) * 1000.0, np.asarray(r.y) * 1000.0
        ok = np.isfinite(x) & np.isfinite(y)
        return x[ok], y[ok]

    for fi, f in enumerate(fields):
        xg, yg = trace_um(f, CHANNEL_WL["G"])
        ref = (xg.mean(), yg.mean())                         # green centroid = image position; R/B offsets = lateral colour
        for ci, ch in enumerate("RGB"):
            x, y = trace_um(f, CHANNEL_WL[ch])
            spot[fi, ci] = float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)))
            k = _hist_kernel(x, y, px_um, ksize, ref)
            s = _diffraction_sigma_px(CHANNEL_WL[ch], fn, px_um)
            gx = np.exp(-0.5 * (np.arange(-3, 4) / s) ** 2)
            k = fftconvolve(k, np.outer(gx, gx) / np.outer(gx, gx).sum(), mode="same")
            ker[fi, ci] = k / max(k.sum(), 1e-12)
    return dict(kernels=ker, fields=fields, spot_rms_um=spot, px_um=px_um)


def mtf_from_kernel(kernel: np.ndarray, px_um: float, freqs_lp_mm):
    """Tangential (y) and sagittal (x) MTF from a PSF kernel: |FFT of the line spread function|."""
    out = {}
    for name, axis in (("tangential", 1), ("sagittal", 0)):
        lsf = kernel.sum(axis=axis)
        F = np.abs(np.fft.rfft(lsf, n=4096))
        F = F / F[0]
        f = np.fft.rfftfreq(4096, d=px_um / 1000.0)   # cycles/mm
        out[name] = np.interp(freqs_lp_mm, f, F).tolist()
    return out


def _azimuth_weights(h, w, n_sec):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.hypot(yy - cy, xx - cx) / (min(h, w) / 2)
    phi = np.arctan2(-(yy - cy), xx - cx)                   # image y points down
    return r, phi


def render_psf(img: np.ndarray, bank: dict, noise_sigma: float = 0.005, seed: int = 0, n_sec: int = 8) -> np.ndarray:
    """img float [0,1] HxWx3 -> degraded. Each pixel gets the PSF of its field radius, rotated to its azimuth."""
    h, w, _ = img.shape
    ker = bank["kernels"]
    r, phi = _azimuth_weights(h, w, n_sec)
    rad_pos = np.clip(r, 0, 1) * (N_FIELD - 1)
    lo = np.clip(np.floor(rad_pos).astype(int), 0, N_FIELD - 2)
    frac = rad_pos - lo
    sec_c = np.arange(n_sec) * 2 * np.pi / n_sec
    dphi = (phi[..., None] - sec_c + np.pi) % (2 * np.pi) - np.pi
    wsec = np.clip(1 - np.abs(dphi) / (2 * np.pi / n_sec), 0, None)
    wsec = wsec / np.maximum(wsec.sum(-1, keepdims=True), 1e-9)
    out = np.zeros_like(img)
    for fi in range(N_FIELD):
        wr = np.where(lo == fi, 1 - frac, 0) + np.where(lo + 1 == fi, frac, 0)
        if not wr.any():
            continue
        for s in range(n_sec):
            wmap = wr * wsec[..., s]
            if wmap.max() < 1e-4:
                continue
            ang = np.degrees(sec_c[s]) - 90.0                # kernels are for a field along +y (phi = 90 deg)
            for c in range(3):
                k = rotate(ker[fi, c], -ang, reshape=False, order=1, mode="constant")
                k = np.clip(k, 0, None)
                k = k / max(k.sum(), 1e-12)
                out[..., c] += wmap * fftconvolve(img[..., c], k, mode="same")
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 1)
