"""Mass, unit-cost and CO2 model, reported RELATIVE to glass = 100 with low/base/high ranges.

ESTIMATED, not measured. The coefficients below are PLACEHOLDER ASSUMPTIONS chosen so the
model structure (per-element grinding/polishing/centering vs. mold amortisation/cycle/yield)
matches Round 1. They are NOT sourced figures; replace them with supplier / literature values
before quoting any cost or CO2 conclusion. Mass uses datasheet densities (BK7/PMMA/PC).
"""
from __future__ import annotations

import numpy as np

DENSITY_G_CM3 = {"N-BK7": 2.51, "N-SF5": 3.86, "PMMA": 1.19, "Polycarbonate": 1.20, "Zeonex": 1.01, "COC_Topas": 1.02, "Polystyrene": 1.05}

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


def unit_cost(kind: str, n_elements: int, yield_base: float | None = None) -> tuple[float, float, float]:
    """(low, base, high). If yield_base is given (e.g. from the tolerance Monte Carlo) it replaces the placeholder yield."""
    out = []
    for i in range(3):
        if kind == "glass":
            y = GLASS_YIELD[i] if yield_base is None else min(1.0, yield_base * (GLASS_YIELD[i] / GLASS_YIELD[1]))
            c = sum(v[_case(i)] for v in GLASS.values()) * n_elements / y
        else:
            y = POLY_YIELD[i] if yield_base is None else min(1.0, yield_base * (POLY_YIELD[i] / POLY_YIELD[1]))
            c = sum(v[_case(i)] for v in POLY.values()) * n_elements / y
        out.append(c)
    return tuple(out)


def monte_carlo_cost(n_poly: int, n_glass: int, poly_yield: float, glass_yield: float = 0.95, n: int = 20000, seed: int = 0) -> dict:
    """Because the coefficients are NOT sourced, test the conclusion against ALL of them at once.

    Every coefficient is drawn uniformly between its low and high value. Reports the share of draws in which the
    polymer lens is cheaper per good unit, the spread of the ratio, which input matters most (Spearman), and the
    break-even polymer yield (median over draws) below which polymer stops being cheaper, and the multiple by which
    ALL polymer costs (tooling, cycle, material...) would have to exceed our assumptions before glass wins. That
    multiple is the honest answer to unsourced coefficients: a claim you can test against a real mold quote.
    ESTIMATED: ranges are assumptions, not supplier quotes.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    names = [f"glass:{k}" for k in GLASS] + [f"poly:{k}" for k in POLY]
    X = np.zeros((n, len(names)))
    ratio, be = np.zeros(n), np.zeros(n)
    for j in range(n):
        g = {k: rng.uniform(v[0], v[2]) for k, v in GLASS.items()}
        p = {k: rng.uniform(v[0], v[2]) for k, v in POLY.items()}
        X[j] = list(g.values()) + list(p.values())
        yg = np.clip(glass_yield + rng.normal(0, 0.03), 0.5, 1)
        yp = np.clip(poly_yield + rng.normal(0, 0.08), 0.2, 1)
        cg = sum(g.values()) * n_glass / yg
        ratio[j] = sum(p.values()) * n_poly / yp / cg
        be[j] = min(1.0, sum(p.values()) * n_poly / cg)  # yield at which the ratio would be exactly 1
    sens = {nm: float(spearmanr(X[:, i], ratio)[0]) for i, nm in enumerate(names)}
    return dict(p_polymer_cheaper=float((ratio < 1).mean()), ratio_p10=float(np.percentile(ratio, 10)),
                ratio_median=float(np.median(ratio)), ratio_p90=float(np.percentile(ratio, 90)),
                break_even_yield_median=float(np.median(be)),
                poly_cost_multiplier_to_break_even=dict(conservative=float(1 / np.percentile(ratio, 90)),
                                                        median=float(1 / np.median(ratio))), sensitivity=dict(sorted(sens.items(), key=lambda kv: -abs(kv[1]))),
                label="Estimated (assumed coefficient ranges, not supplier quotes)")


def mount_mass_g(kind: str, n_elements: int, rng) -> float:
    """Spacers/retainers/barrel: PLACEHOLDER ranges. Glass stacks need one retainer per element; molded parts can integrate the mount."""
    return float(n_elements * (rng.uniform(0.2, 0.8) if kind == "glass" else rng.uniform(0.05, 0.25)))


def co2(kind: str, mass_g: float) -> tuple[float, float, float]:
    return tuple(CO2_PER_KG[kind][i] * mass_g / 1000.0 for i in range(3))


def relative(values: tuple, glass: tuple) -> tuple[float, float, float]:
    """Express a (low, base, high) triple against the glass base case = 100, keeping case pairing."""
    return tuple(100.0 * v / glass[1] for v in values)


def conclusion_flips(poly_rel: tuple, glass_rel_base: float = 100.0) -> bool:
    """True if polymer is cheaper than glass in some cases but not others (Round 1 requirement)."""
    below = [v < glass_rel_base for v in poly_rel]
    return any(below) and not all(below)
