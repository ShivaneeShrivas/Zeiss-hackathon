"""Optical models in Optiland (open-source differentiable ray tracer). Everything here is SIMULATED.

Element dict: {r1, r2, t, gap, material, [k1, a1, k2, a2]}  (k = conic, a = [A4, A6, A8] even-asphere terms).
The first surface is the stop; the last element's gap is replaced by the back-focus distance.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from optiland.analysis import SpotDiagram  # noqa: E402
from optiland.materials import Material  # noqa: E402
from optiland.optic import Optic  # noqa: E402
from optiland.optimization import LeastSquares, OptimizationProblem  # noqa: E402

WAVELENGTHS_UM = (0.4861, 0.5876, 0.6563)  # F, d, C lines; d primary
FIELDS = (0.0, 0.5, 1.0)                    # fraction of half-FOV
FIELD_NAMES = ["on-axis", "50% field", "edge"]


def build_lens(elements, epd_mm, half_fov_deg, back_mm=30.0, fields=FIELDS):
    lens = Optic()
    lens.surfaces.add(index=0, thickness=np.inf)
    idx = 1
    for k, el in enumerate(elements):
        gap = back_mm if k == len(elements) - 1 else el.get("gap", 1.0)
        for s, (r, th, mat) in enumerate(((el["r1"], el["t"], el["material"]), (el["r2"], gap, None))):
            kw = dict(index=idx + s, radius=r, thickness=th)
            if mat:
                from frugaloptix.materials import resolve
                kw["material"] = resolve(mat)
            if idx + s == 1:
                kw["is_stop"] = True
            kk, aa = el.get(f"k{s + 1}"), el.get(f"a{s + 1}")
            if kk is not None or aa is not None:
                kw.update(surface_type="even_asphere", conic=float(kk or 0.0), coefficients=[0.0] + list(aa or [0, 0, 0]))
            lens.surfaces.add(**kw)
        idx += 2
    lens.surfaces.add(index=idx)
    lens.set_aperture(aperture_type="EPD", value=epd_mm)
    lens.fields.set_type("angle")
    for f in fields:
        lens.fields.add(y=f * half_fov_deg)
    for i, w in enumerate(WAVELENGTHS_UM):
        lens.wavelengths.add(value=w, is_primary=(i == 1))
    return lens


def rms_spot_um(lens):
    """RMS spot radius (um) per field, averaged over wavelengths."""
    rms = np.array(SpotDiagram(lens).rms_spot_radius())
    return (rms.mean(axis=1) * 1000.0).tolist()


def _spot_operands(prob, lens, weights):
    for fi, hy in enumerate(FIELDS):
        for w in WAVELENGTHS_UM:
            prob.add_operand(operand_type="rms_spot_size", target=0.0, weight=float(weights[fi]),
                             input_data={"optic": lens, "surface_number": -1, "Hx": 0.0, "Hy": hy, "num_rays": 5,
                                         "wavelength": w, "distribution": "hexapolar"})


def optimise(lens, target_efl_mm, weights=(1, 1, 1), max_iter=60, aspheric_surfaces=(), radii=True):
    """Least-squares over radii, back focus and (optionally) conic + A4/A6/A8 of the listed surfaces."""
    prob = OptimizationProblem()
    n = lens.surfaces.num_surfaces
    if radii:
        for s in range(1, n - 1):
            prob.add_variable(lens, "radius", surface_number=s, min_val=-300, max_val=300)
    prob.add_variable(lens, "thickness", surface_number=n - 2, min_val=5, max_val=80)
    for s in aspheric_surfaces:
        prob.add_variable(lens, "conic", surface_number=s, min_val=-5, max_val=5)
        for c in (1, 2, 3):
            prob.add_variable(lens, "asphere_coeff", surface_number=s, coeff_number=c, min_val=-0.01, max_val=0.01)
    if radii:
        prob.add_operand(operand_type="f2", target=target_efl_mm, weight=5.0, input_data={"optic": lens})
    _spot_operands(prob, lens, weights)
    LeastSquares(prob).optimise(maxiter=max_iter) if hasattr(LeastSquares, "optimise") else LeastSquares(prob).optimize(maxiter=max_iter)
    return lens


def refocus(lens, max_iter=25):
    """Only re-adjust back focus (used for element ablation and thermal checks)."""
    return optimise(lens, 0.0, max_iter=max_iter, radii=False)


def read_back(lens, elements):
    """Optimised prescription as plain dicts (radii, back focus, conic/asphere terms)."""
    radii = np.asarray(lens.surfaces.radii, dtype=float)
    out = []
    for k, el in enumerate(elements):
        e = dict(el)
        for s in (1, 2):
            si = 2 * k + s
            e[f"r{s}"] = float(radii[si])
            g = lens.surfaces.surfaces[si].geometry
            if hasattr(g, "k") and (el.get(f"k{s}") is not None or el.get(f"a{s}") is not None):
                e[f"k{s}"] = float(np.asarray(g.k))
                e[f"a{s}"] = [float(np.asarray(c)) for c in list(g.coefficients)[1:4]]
        out.append(e)
    back = float(np.asarray(lens.surfaces.get_thickness(lens.surfaces.num_surfaces - 2)).ravel()[0])
    return out, back


def semi_apertures_mm(lens):
    n = lens.surfaces.num_surfaces
    h = np.zeros(n)
    for hy in FIELDS:
        lens.trace(Hx=0.0, Hy=hy, wavelength=WAVELENGTHS_UM[1], num_rays=24, distribution="hexapolar")
        h = np.maximum(h, np.nanmax(np.abs(np.asarray(lens.surfaces.y)), axis=1))
    return (1.05 * h).tolist()


def element_volume_mm3(r1, r2, t, semi_ap, k1=0.0, a1=None, k2=0.0, a2=None, n_r=400):
    r = np.linspace(0.0, semi_ap, n_r)

    def sag(R, k, a):
        if not np.isfinite(R):
            z = np.zeros_like(r)
        else:
            z = r ** 2 / (R * (1 + np.sqrt(np.maximum(1 - (1 + k) * r ** 2 / R ** 2, 1e-9))))
        for i, c in enumerate(a or [], start=2):  # A4 r^4, A6 r^6, A8 r^8
            z = z + c * r ** (2 * i)
        return z

    prof = t + sag(r2, k2 or 0.0, a2) - sag(r1, k1 or 0.0, a1)
    return float(np.trapezoid(2 * np.pi * r * np.maximum(prof, 0.0), r))


def evaluate(design, spec):
    """Build a design dict {elements, back_mm} -> RMS spots, mass and EFL."""
    from frugaloptix import costmodel as cm
    lens = build_lens(design["elements"], spec["epd_mm"], spec["half_fov_deg"], design["back_mm"])
    spot = rms_spot_um(lens)
    semi = semi_apertures_mm(lens)[1:-1]
    mass = 0.0
    for k, e in enumerate(design["elements"]):
        v = element_volume_mm3(e["r1"], e["r2"], e["t"], max(semi[2 * k], semi[2 * k + 1]), e.get("k1"), e.get("a1"), e.get("k2"), e.get("a2"))
        mass += v / 1000 * cm.DENSITY_G_CM3[e["material"]]
    try:
        efl = float(np.asarray(lens.paraxial.f2()).ravel()[0])
    except Exception:
        efl = float("nan")
    return dict(spot_um=spot, mass_g=mass, efl_mm=efl)


# ------------------ tolerance Monte Carlo (molding / grinding errors) ------------------
TOL_GRADES = {  # ASSUMED process grades (1-sigma). Estimated, not sourced: replace with supplier data.
    "glass_ground": dict(r_rel=0.0005, t_mm=0.010, a_rel=0.0),
    "polymer_molded": dict(r_rel=0.003, t_mm=0.030, a_rel=0.02),
}


def perturb(design, grade, rng, scale=1.0):
    d = {"elements": [], "back_mm": design["back_mm"]}
    for e in design["elements"]:
        e2 = dict(e)
        e2["r1"] = e["r1"] * (1 + rng.normal(0, grade["r_rel"] * scale))
        e2["r2"] = e["r2"] * (1 + rng.normal(0, grade["r_rel"] * scale))
        e2["t"] = max(0.5, e["t"] + rng.normal(0, grade["t_mm"] * scale))
        e2["gap"] = max(0.2, e.get("gap", 1.0) + rng.normal(0, grade["t_mm"] * scale))
        for s in (1, 2):
            if e.get(f"a{s}"):
                e2[f"a{s}"] = [c * (1 + rng.normal(0, grade["a_rel"] * scale)) for c in e[f"a{s}"]]
        d["elements"].append(e2)
    return d


def monte_carlo(design, spec, grade, n, floor_um, seed=0, scale=1.0, refocus_each=False):
    """Returns (yield fraction, array of spot sets). A draw passes if every field's RMS spot <= floor."""
    rng = np.random.default_rng(seed)
    spots = []
    for _ in range(n):
        d = perturb(design, grade, rng, scale)
        lens = build_lens(d["elements"], spec["epd_mm"], spec["half_fov_deg"], d["back_mm"])
        if refocus_each:
            refocus(lens, 15)
        spots.append(rms_spot_um(lens))
    spots = np.array(spots)
    ok = np.all(spots <= np.array(floor_um)[None, :], axis=1)
    return float(ok.mean()), spots


# ------------------ thermal defocus (literature-typical coefficients, ESTIMATED) ------------------
THERMAL = {  # (dn/dT per K, linear expansion per K): order-of-magnitude typical values, VERIFY against datasheets
    "N-BK7": (3e-6, 7.1e-6), "N-SF5": (4e-6, 8.3e-6), "PMMA": (-0.85e-4, 7e-5),
    "Polycarbonate": (-1.18e-4, 6.5e-5), "Zeonex": (-0.8e-4, 6e-5),
}


def thermal_defocus_mm(design, dT):
    """Back-focus shift for a temperature change dT (K), thin-lens power sum; housing expansion ignored."""
    phi_tot, dphi = 0.0, 0.0
    for e in design["elements"]:
        from frugaloptix.materials import DB, resolve
        n = DB[e["material"]]["nd"] if e["material"] in DB else float(np.asarray(Material(e["material"]).n(0.5876)).ravel()[0])
        phi = (n - 1) * (1 / e["r1"] - 1 / e["r2"])
        dn, al = (THERMAL[e["material"]] if e["material"] in THERMAL else __import__("frugaloptix.materials", fromlist=["thermal"]).thermal(e["material"]))
        phi_tot += phi
        dphi += -phi * (al - dn / (n - 1)) * dT
    return float(-dphi / phi_tot ** 2) if phi_tot else 0.0


def defocus_rms_um(dz_mm, f_number):
    """Geometric blur-circle RMS radius from a focus error (diameter = dz/N, rms radius ~ diameter/4)."""
    return abs(dz_mm) / f_number / 4.0 * 1000.0
