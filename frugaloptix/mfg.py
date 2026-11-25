"""Manufacturing-constraint checks for a lens design (hard gates, thresholds from configs/acceptance.yaml)."""
from __future__ import annotations

import numpy as np
import yaml

from frugaloptix import optics as op
from frugaloptix.ml import ROOT


def limits():
    return yaml.safe_load((ROOT / "configs" / "acceptance.yaml").read_text())["manufacturing"]


def _sag(r, R, k, a):
    z = np.zeros_like(r) if not np.isfinite(R) else r ** 2 / (R * (1 + np.sqrt(np.maximum(1 - (1 + (k or 0.0)) * r ** 2 / R ** 2, 1e-9))))
    for i, c in enumerate(a or [], start=2):
        z = z + c * r ** (2 * i)
    return z


def check(design: dict, spec: dict, lim: dict | None = None) -> dict:
    lim = lim or limits()
    lens = op.build_lens(design["elements"], spec["epd_mm"], spec["half_fov_deg"], design["back_mm"])
    semi = op.semi_apertures_mm(lens)[1:-1]
    rows, ok_all = [], True
    for k, e in enumerate(design["elements"]):
        sa = max(semi[2 * k], semi[2 * k + 1])
        r = np.linspace(0, sa, 300)
        s1, s2 = _sag(r, e["r1"], e.get("k1"), e.get("a1")), _sag(r, e["r2"], e.get("k2"), e.get("a2"))
        thick = e["t"] + s2 - s1
        slope = lambda R, kk, a: np.degrees(np.arctan(np.abs(np.gradient(_sag(r, R, kk, a), r))))  # noqa: E731
        dep = lambda R, kk, a: float(np.max(np.abs(_sag(r, R, kk, a) - _sag(r, R, 0.0, None))))      # noqa: E731
        gap = e.get("gap", 1.0) if k < len(design["elements"]) - 1 else None
        row = dict(element=k + 1, material=e["material"], semi_aperture_mm=float(sa), centre_thickness_mm=float(e["t"]),
                   edge_thickness_mm=float(thick[-1]), min_thickness_mm=float(thick.min()),
                   thickness_ratio=float(thick.max() / max(thick.min(), 1e-6)),
                   max_slope_deg=float(max(slope(e["r1"], e.get("k1"), e.get("a1")).max(), slope(e["r2"], e.get("k2"), e.get("a2")).max())),
                   asphere_departure_mm=float(max(dep(e["r1"], e.get("k1"), e.get("a1")), dep(e["r2"], e.get("k2"), e.get("a2")))), air_gap_after_mm=gap)
        v = []
        if row["centre_thickness_mm"] < lim["min_centre_thickness_mm"]: v.append("centre thickness")
        if row["min_thickness_mm"] < lim["min_edge_thickness_mm"]: v.append("edge thickness")
        if gap is not None and gap < lim["min_air_gap_mm"]: v.append("air gap")
        if row["max_slope_deg"] > lim["max_surface_slope_deg"]: v.append("surface slope")
        if row["asphere_departure_mm"] > lim["max_asphere_departure_mm"]: v.append("asphere departure")
        if row["thickness_ratio"] > lim["max_thickness_ratio"]: v.append("thickness ratio")
        row["violations"] = v
        ok_all &= not v
        rows.append(row)
    return dict(pass_=bool(ok_all), elements=rows)
