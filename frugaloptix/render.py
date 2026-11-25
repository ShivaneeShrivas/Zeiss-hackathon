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


# ---------------------------------------------------------------------------------------------
# Torch renderer used to generate RANDOMISED degradations on the fly while training (no fixed blur to memorise)
# ---------------------------------------------------------------------------------------------
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _gblur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.25:
        return x
    r = max(1, int(3 * sigma + 0.5))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / sigma) ** 2)
    k = (k / k.sum())
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _zone_weights(h, w, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    rr = torch.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (min(h, w) / 2)
    wts = torch.stack([torch.clamp(1 - (rr - f).abs() / 0.5, 0, 1) for f in (0.0, 0.5, 1.0)])
    return wts / (wts.sum(0, keepdim=True) + 1e-9)


def degrade_torch(x: torch.Tensor, spots_um, px_um: float = 10.0, noise_sigma=0.005) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1]; spots_um: (B,3) RMS spot radii at field 0/.5/1; noise_sigma: float or (B,) tensor."""
    spots = torch.as_tensor(np.asarray(spots_um), dtype=x.dtype, device=x.device)
    wz = _zone_weights(x.shape[2], x.shape[3], x.device)
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        layers = torch.stack([_gblur(x[i:i + 1], float(max(spots[i, f] / px_um, 1e-3)))[0] for f in range(3)])
        out[i] = (layers * wz[:, None]).sum(0)
    ns = torch.as_tensor(noise_sigma, dtype=x.dtype, device=x.device).reshape(-1, 1, 1, 1) if not isinstance(noise_sigma, float) else noise_sigma
    return (out + torch.randn_like(out) * ns).clamp(0, 1)


class DegradationSampler:
    """Training-time degradations: tolerance Monte Carlo spot sets + thermal defocus + varying sensor noise.

    This is how 'molding errors and temperature shifts are injected in training' (Round 1) is implemented.
    """

    def __init__(self, nominal_spots, mc_spots=None, thermal_rms_um=0.0, seed=0):
        self.nominal = np.asarray(nominal_spots, dtype=float)
        self.mc = None if mc_spots is None else np.asarray(mc_spots, dtype=float)
        self.thermal = float(thermal_rms_um)
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        if self.mc is not None and len(self.mc):
            s = self.mc[self.rng.integers(0, len(self.mc), n)] * self.rng.uniform(0.9, 1.1, (n, 1))
        else:
            s = self.nominal[None] * self.rng.uniform(0.7, 1.5, (n, 1))
        extra = self.rng.uniform(0, self.thermal, (n, 1))
        s = np.sqrt(s ** 2 + extra ** 2)
        return s, self.rng.uniform(0.002, 0.02, n)


# ---------------------------------------------------------------------------------------------
# Independent 'different physics' renderer for out-of-distribution testing (NEVER used in training):
# heavy-tailed Moffat PSF, lateral colour, vignetting, signal-dependent (Poisson-Gaussian) noise.
# ---------------------------------------------------------------------------------------------
def _moffat_kernel(sigma_px: float, beta: float = 2.5) -> np.ndarray:
    fwhm = 2.355 * max(sigma_px, 0.3)
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    r = int(min(max(4 * fwhm, 3), 40))
    ax = np.arange(-r, r + 1)
    k = (1 + (ax[:, None] ** 2 + ax[None] ** 2) / alpha ** 2) ** (-beta)
    return k / k.sum()


def render_ood(img: np.ndarray, rms_um_by_field, px_um: float = 10.0, seed: int = 0) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    from scipy.signal import fftconvolve
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    rr = np.hypot(yy - cy, xx - cx) / (min(h, w) / 2)
    wts = np.stack([np.clip(1 - np.abs(rr - f) / 0.5, 0, 1) for f in (0.0, 0.5, 1.0)])
    wts = wts / (wts.sum(0) + 1e-9)
    out = np.zeros_like(img, dtype=float)
    for f, s in enumerate(rms_um_by_field):
        k = _moffat_kernel(max(s / px_um, 0.3))
        layer = np.stack([fftconvolve(img[..., c], k, mode="same") for c in range(img.shape[2])], -1)
        out += wts[f][..., None] * layer
    shift = 1.0 / (min(h, w) / 2)  # ~1 px lateral colour at the edge: red magnified, blue shrunk
    for c, sc in ((0, 1 + shift), (2, 1 - shift)):
        out[..., c] = map_coordinates(out[..., c], [(yy - cy) / sc + cy, (xx - cx) / sc + cx], order=1, mode="nearest")
    out *= (1 - 0.25 * rr ** 2)[..., None]  # vignetting
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0, 1, out.shape) * np.sqrt(np.clip(out, 0, 1) * 0.004 + 1e-6) * 0.5 + rng.normal(0, 0.004, out.shape)
    return np.clip(out, 0, 1)
