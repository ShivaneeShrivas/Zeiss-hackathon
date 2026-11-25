"""Tolerance Monte Carlo with radii, thickness/air-gap, asphere-term, DECENTER and TILT errors, Wilson CIs and a
one-at-a-time sensitivity ranking. Grades: sigma = published limit / 2 (limits from Apollo Optical's injection-moulding table,
https://www.apollooptical.com/manufacturing-tolerances ; glass grades and tilt ASSUMED). Refractive-index variation is NOT modelled."""
from __future__ import annotations

import numpy as np

from frugaloptix import optics as op

GRADES = {  # 1-sigma values
    "polymer_commercial": dict(r_rel=0.005, t_mm=0.0125, dec_mm=0.0125, tilt_deg=0.025, a_rel=0.02,
                               source="Apollo Optical commercial: radius +-1%, CT +-0.025 mm, concentricity 0.025 mm (limit = 2 sigma); tilt, asphere ASSUMED"),
    "polymer_precision": dict(r_rel=0.0025, t_mm=0.0075, dec_mm=0.0075, tilt_deg=0.0125, a_rel=0.01,
                              source="Apollo Optical precision: radius +-0.5%, CT +-0.015 mm, concentricity 0.015 mm (limit = 2 sigma); tilt, asphere ASSUMED"),
    "glass_ground": dict(r_rel=0.0005, t_mm=0.01, dec_mm=0.005, tilt_deg=0.005, a_rel=0.0,
                         source="ASSUMED typical precision ground/polished glass; replace with supplier data"),
}
GROUPS = {"radius": ("r_rel",), "thickness_gap": ("t_mm",), "asphere": ("a_rel",), "decenter": ("dec_mm",), "tilt": ("tilt_deg",)}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def draw(design, grade, rng, active=None):
    """-> (perturbed design, per-element (dec_y_mm, tilt_rad)). `active` limits which error groups are applied."""
    act = set(sum((GROUPS[g] for g in (active or GROUPS)), ()))
    g = {k: (v if k in act else 0.0) for k, v in grade.items() if k != "source"}
    d = {"elements": [], "back_mm": design["back_mm"]}
    pose = []
    for e in design["elements"]:
        e2 = dict(e)
        e2["r1"] = e["r1"] * (1 + rng.normal(0, g["r_rel"]))
        e2["r2"] = e["r2"] * (1 + rng.normal(0, g["r_rel"]))
        e2["t"] = max(0.5, e["t"] + rng.normal(0, g["t_mm"]))
        e2["gap"] = max(0.2, e.get("gap", 1.0) + rng.normal(0, g["t_mm"]))
        for s in (1, 2):
            if e.get(f"a{s}"):
                e2[f"a{s}"] = [c * (1 + rng.normal(0, g["a_rel"])) for c in e[f"a{s}"]]
        d["elements"].append(e2)
        pose.append((rng.normal(0, g["dec_mm"]), np.radians(rng.normal(0, g["tilt_deg"]))))
    return d, pose


def spots(design, spec, pose):
    lens = op.build_lens(design["elements"], spec["epd_mm"], spec["half_fov_deg"], design["back_mm"])
    for k, (dy, tl) in enumerate(pose):
        for s in (2 * k + 1, 2 * k + 2):
            cs = lens.surfaces.surfaces[s].geometry.cs
            cs.y = cs.y + dy
            cs.rx = cs.rx + tl
    return op.rms_spot_um(lens)


def run(design, spec, grade, n, seed=0, active=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d, pose = draw(design, grade, rng, active)
        out.append(spots(d, spec, pose))
    return np.array(out)


def yield_ci(sp, limit_um):
    ok = np.all(sp <= np.asarray(limit_um)[None, :], axis=1)
    return dict(yield_=float(ok.mean()), lo=wilson(int(ok.sum()), len(ok))[0], hi=wilson(int(ok.sum()), len(ok))[1], n=int(len(ok)))


def sensitivity(design, spec, grade, n=120, seed=1):
    """Mean increase of the worst-field RMS spot when ONE error group acts alone (others off)."""
    nom = max(op.rms_spot_um(op.build_lens(design["elements"], spec["epd_mm"], spec["half_fov_deg"], design["back_mm"])))
    res = {}
    for gname in GROUPS:
        sp = run(design, spec, grade, n, seed, active=[gname])
        w = sp.max(axis=1)
        res[gname] = dict(mean_increase_um=float(w.mean() - nom), p95_increase_um=float(np.percentile(w, 95) - nom))
    return dict(nominal_worst_um=float(nom), by_group=res)
