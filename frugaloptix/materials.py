"""Optical-polymer database. nd/Vd/dn/dT/birefringence/Tmax from Apollo Optical's polymer table
(https://www.apollooptical.com/common-polymers-for-optical-components); water absorption, CTE and density are
TYPICAL DATASHEET values (Estimated: confirm with the supplier grade you actually buy). Optiland catalog is used where it has the material."""
from __future__ import annotations

import numpy as np

# name: nd, Vd, dndT (1e-5/K), cte (1e-5/K), water_abs_pct, density, birefringence (1 low .. 3 high), tmax_C, scratch(1 poor..3 good), uv(1 poor..3 good), source
DB = {
    "PMMA":          dict(nd=1.491, vd=57.4, dndt=-8.5, cte=7.0, water=0.30, rho=1.19, biref=1, tmax=80, scratch=3, uv=3, catalog="PMMA"),
    "Polycarbonate": dict(nd=1.585, vd=29.9, dndt=-11.8, cte=6.5, water=0.15, rho=1.20, biref=3, tmax=120, scratch=1, uv=1, catalog="Polycarbonate"),
    "Zeonex":        dict(nd=1.530, vd=55.8, dndt=-8.0, cte=6.0, water=0.01, rho=1.01, biref=1, tmax=120, scratch=2, uv=2, catalog="Zeonex"),  # COP
    "COC_Topas":     dict(nd=1.533, vd=56.0, dndt=-10.1, cte=6.0, water=0.01, rho=1.02, biref=1, tmax=105, scratch=2, uv=2, catalog=None),    # COC, custom Abbe model
    "Polystyrene":   dict(nd=1.590, vd=30.9, dndt=-14.0, cte=7.0, water=0.05, rho=1.05, biref=3, tmax=75, scratch=1, uv=1, catalog=None),
}
CUSTOM = {k for k, v in DB.items() if v["catalog"] is None}


def resolve(name: str):
    """Name -> something Optiland accepts as a material."""
    if name in CUSTOM:
        from optiland.materials import AbbeMaterial
        return AbbeMaterial(n=DB[name]["nd"], abbe=DB[name]["vd"])
    return name


def thermal(name: str):
    d = DB[name]
    return (d["dndt"] * 1e-5, d["cte"] * 1e-5)


def suitability(name: str, tmax_needed=70.0) -> dict:
    """Transparent non-optical screen (hard filters + a 0..1 score). Optical score is added elsewhere."""
    d = DB[name]
    fails = []
    if d["tmax"] < tmax_needed: fails.append(f"Tmax {d['tmax']} C < {tmax_needed} C (car-boot / sunlight)")
    if d["biref"] >= 3: fails.append("high stress birefringence (hurts fundus contrast)")
    score = (0.30 * (1 - min(d["water"], 0.4) / 0.4) + 0.25 * (4 - d["biref"]) / 3 + 0.15 * (d["scratch"] - 1) / 2
             + 0.15 * (d["uv"] - 1) / 2 + 0.15 * min(d["tmax"], 120) / 120)
    return dict(material=name, score=float(score), hard_fail=fails, **{k: d[k] for k in ("nd", "vd", "dndt", "water", "biref", "tmax")})
