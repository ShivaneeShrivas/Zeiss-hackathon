"""Mass, unit-cost and CO2 model, reported RELATIVE to glass = 100 with low/base/high ranges.

ESTIMATED, not measured. The coefficients below are PLACEHOLDER ASSUMPTIONS chosen so the
model structure (per-element grinding/polishing/centering vs. mold amortisation/cycle/yield)
matches Round 1. They are NOT sourced figures; replace them with supplier / literature values
before quoting any cost or CO2 conclusion. Mass uses datasheet densities (BK7/PMMA/PC).
"""
from __future__ import annotations

DENSITY_G_CM3 = {"N-BK7": 2.51, "N-SF5": 3.86, "PMMA": 1.19, "Polycarbonate": 1.20}

# (low, base, high) PLACEHOLDER coefficients -- see module docstring
GLASS = {  # per element, arbitrary currency units
    "blank": (0.8, 1.2, 2.0), "grind_polish": (2.5, 4.0, 6.0), "centering": (0.8, 1.2, 1.8),
    "assembly": (0.5, 0.8, 1.2), "coating": (0.3, 0.5, 0.8),
}
POLY = {  # per part
    "material": (0.05, 0.08, 0.15), "cycle": (0.3, 0.5, 0.9), "mold_amort": (0.2, 0.4, 0.8),
    "assembly": (0.3, 0.5, 0.8), "coating": (0.3, 0.5, 0.8),
}
POLY_YIELD = (0.95, 0.85, 0.70)   # share of good parts (high-cost case = low yield)
GLASS_YIELD = (0.98, 0.95, 0.90)
CO2_PER_KG = {"glass": (1.0, 1.5, 2.5), "polymer": (2.5, 3.5, 5.0)}  # kg CO2e / kg part, placeholder


def _case(i: int) -> int:
    return i  # 0=low, 1=base, 2=high


def unit_cost(kind: str, n_elements: int) -> tuple[float, float, float]:
    out = []
    for i in range(3):
        if kind == "glass":
            c = sum(v[_case(i)] for v in GLASS.values()) * n_elements / GLASS_YIELD[i]
        else:
            c = sum(v[_case(i)] for v in POLY.values()) * n_elements / POLY_YIELD[i]
        out.append(c)
    return tuple(out)


def co2(kind: str, mass_g: float) -> tuple[float, float, float]:
    return tuple(CO2_PER_KG[kind][i] * mass_g / 1000.0 for i in range(3))


def relative(values: tuple, glass: tuple) -> tuple[float, float, float]:
    """Express a (low, base, high) triple against the glass base case = 100, keeping case pairing."""
    return tuple(100.0 * v / glass[1] for v in values)


def conclusion_flips(poly_rel: tuple, glass_rel_base: float = 100.0) -> bool:
    """True if polymer is cheaper than glass in some cases but not others (Round 1 requirement)."""
    below = [v < glass_rel_base for v in poly_rel]
    return any(below) and not all(below)
