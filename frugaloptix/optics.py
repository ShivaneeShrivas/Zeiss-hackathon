"""Optical baselines built in Optiland (open-source, differentiable ray tracer).

Everything here is SIMULATED: numbers come from ray tracing a simplified
imaging objective, not from a physical lens. See configs/requirement_sheet.yaml.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from optiland.analysis import SpotDiagram  # noqa: E402
from optiland.optic import Optic  # noqa: E402
from optiland.optimization import (  # noqa: E402
    LeastSquares,
    OptimizationProblem,
    RadiusVariable,
)

WAVELENGTHS_UM = (0.4861, 0.5876, 0.6563)  # F, d, C lines; d (green-ish) primary


def build_lens(elements: list[dict], epd_mm: float, half_fov_deg: float, back_mm: float = 30.0) -> Optic:
    """elements: [{r1, r2, t, material}] listed object->image. First surface is the stop."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=np.inf)
    idx = 1
    for k, el in enumerate(elements):
        lens.surfaces.add(index=idx, radius=el["r1"], thickness=el["t"], material=el["material"],
                          is_stop=(k == 0))
        lens.surfaces.add(index=idx + 1, radius=el["r2"], thickness=(back_mm if k == len(elements) - 1 else el.get("gap", 1.0)))
        idx += 2
    lens.surfaces.add(index=idx)  # image plane
    lens.set_aperture(aperture_type="EPD", value=epd_mm)
    lens.fields.set_type("angle")
    for f in (0.0, 0.5, 1.0):
        lens.fields.add(y=f * half_fov_deg)
    for i, w in enumerate(WAVELENGTHS_UM):
        lens.wavelengths.add(value=w, is_primary=(i == 1))
    return lens


def rms_spot_um(lens: Optic) -> list[float]:
    """RMS spot radius (um) at each field, averaged over wavelengths. SIMULATED."""
    sd = SpotDiagram(lens)
    rms = np.array(sd.rms_spot_radius())  # [field][wavelength], mm
    return (rms.mean(axis=1) * 1000.0).tolist()


def optimise_radii(lens: Optic, target_efl_mm: float, max_iter: int = 60) -> Optic:
    """Least-squares optimisation of all surface radii for RMS spot + fixed EFL."""
    prob = OptimizationProblem()
    n = lens.surfaces.num_surfaces
    for s in range(1, n - 1):
        prob.add_variable(lens, "radius", surface_number=s, min_val=-300, max_val=300)
    prob.add_variable(lens, "thickness", surface_number=n - 2, min_val=5, max_val=80)  # back focus
    prob.add_operand(operand_type="f2", target=target_efl_mm, weight=5.0, input_data={"optic": lens})
    for hy in (0.0, 0.5, 1.0):
        for w in WAVELENGTHS_UM:
            prob.add_operand(
                operand_type="rms_spot_size", target=0.0, weight=1.0,
                input_data={"optic": lens, "surface_number": -1, "Hx": 0.0, "Hy": hy,
                            "num_rays": 5, "wavelength": w, "distribution": "hexapolar"},
            )
    LeastSquares(prob).optimize(maxiter=max_iter)
    return lens


def read_back(lens: Optic, elements: list[dict]) -> tuple[list[dict], float]:
    """Return elements with optimised radii plus the optimised back-focus distance (mm)."""
    radii = np.asarray(lens.surfaces.radii, dtype=float)
    out = []
    for k, el in enumerate(elements):
        e = dict(el)
        e["r1"], e["r2"] = float(radii[1 + 2 * k]), float(radii[2 + 2 * k])
        out.append(e)
    back = float(np.asarray(lens.surfaces.get_thickness(lens.surfaces.num_surfaces - 2)).ravel()[0])
    return out, back


def element_volume_mm3(r1: float, r2: float, t: float, semi_ap: float, n_r: int = 400) -> float:
    """Volume of a meniscus/bi-convex singlet from its sag profile (numerical, mm^3)."""
    r = np.linspace(0.0, semi_ap, n_r)

    def sag(R):
        if not np.isfinite(R):
            return np.zeros_like(r)
        return R - np.sign(R) * np.sqrt(np.maximum(R * R - r * r, 0.0))

    thickness_profile = t + sag(r2) - sag(r1)
    return float(np.trapezoid(2 * np.pi * r * np.maximum(thickness_profile, 0.0), r))


def semi_apertures_mm(lens: Optic) -> list[float]:
    """Clear semi-aperture per surface = max ray height over fields (with a 5% margin)."""
    n = lens.surfaces.num_surfaces
    h = np.zeros(n)
    for hy in (0.0, 0.5, 1.0):
        lens.trace(Hx=0.0, Hy=hy, wavelength=WAVELENGTHS_UM[1], num_rays=24, distribution="hexapolar")
        h = np.maximum(h, np.nanmax(np.abs(np.asarray(lens.surfaces.y)), axis=1))
    return (1.05 * h).tolist()
