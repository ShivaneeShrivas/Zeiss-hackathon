"""Stages 2, 3, 5 and the optical half of 4: everything that needs only the ray tracer.

1. Material swap: aspheric 2-element polymer designs for several material pairs (incl. single-material = recyclable),
   optimised with saliency-derived field weights (results/saliency.json if present).
2. Design X-ray: remove each glass element, refocus, see what the image loses vs what the element costs.
3. Tolerance Monte Carlo: yield vs process-grade scale for glass and molded polymer, loosest passing grade.
4. Thermal defocus 15-35 C (ESTIMATED), mass, cost/CO2 with Monte-Carlo-derived yield, 3-element fallback.
Writes configs/designs.json, results/optics.json, results/design_study.json, results/mc_spots.npy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frugaloptix import costmodel as cm  # noqa: E402
from frugaloptix import optics as op  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mc", type=int, default=40, help="Monte Carlo draws per grade")
ap.add_argument("--iters", type=int, default=80)
ap.add_argument("--quick", action="store_true", help="fewer material pairs / draws (for smoke tests)")
a = ap.parse_args()
SPEC = yaml.safe_load((ROOT / "configs" / "requirement_sheet.yaml").read_text())
DES = json.loads((ROOT / "configs" / "designs.json").read_text())
sal = ROOT / "results" / "saliency.json"
weights = json.loads(sal.read_text())["field_weights"] if sal.exists() else [1.0, 1.0, 1.0]
print("field weights used for the polymer design:", weights, "(from saliency)" if sal.exists() else "(uniform: run 02b_saliency.py first)")
t0 = time.time()
glass = dict(elements=DES["glass_4el"]["elements"], back_mm=DES["glass_4el"]["back_mm"])
g_ev = op.evaluate(glass, SPEC)
floor = [2.0 * s for s in g_ev["spot_um"]]  # declared up front: quality floor = 2x glass nominal RMS spot at each field
wmean = lambda spots: float(np.average(spots, weights=weights))  # noqa: E731

# ---------- 1. material swap (aspheric 2-element) ----------
sph = DES["polymer_2el"]
combos = [("PMMA", "Polycarbonate"), ("PMMA", "PMMA"), ("Zeonex", "Polycarbonate"), ("Zeonex", "Zeonex"), ("PMMA", "Zeonex")]
if a.quick:
    combos = combos[:2]
cands = []
for m1, m2 in combos:
    els = [dict(e, material=m, k1=0.0, a1=[0, 0, 0], k2=0.0, a2=[0, 0, 0]) for e, m in zip(sph["elements"], (m1, m2))]
    lens = op.build_lens(els, SPEC["epd_mm"], SPEC["half_fov_deg"], sph["back_mm"])
    op.optimise(lens, SPEC["efl_mm"], weights, a.iters, aspheric_surfaces=(1, 2, 3, 4))
    e2, back = op.read_back(lens, els)
    d = dict(elements=e2, back_mm=back)
    ev = op.evaluate(d, SPEC)
    kind_cost = cm.unit_cost("polymer", 2)
    cands.append(dict(name=f"{m1}+{m2}", design=d, spot_um=ev["spot_um"], mass_g=ev["mass_g"], weighted_spot_um=wmean(ev["spot_um"]),
                      single_material=(m1 == m2), cost_base=kind_cost[1], co2_base=cm.co2("polymer", ev["mass_g"])[1]))
    print(f"  {m1}+{m2}: spots {[round(x, 1) for x in ev['spot_um']]} mass {ev['mass_g']:.2f} g  ({time.time() - t0:.0f}s)")
cands.sort(key=lambda c: c["weighted_spot_um"])
for i, c in enumerate(cands):  # Pareto: not dominated in (weighted spot, mass)
    c["pareto"] = not any(o["weighted_spot_um"] <= c["weighted_spot_um"] and o["mass_g"] <= c["mass_g"] and
                          (o["weighted_spot_um"] < c["weighted_spot_um"] or o["mass_g"] < c["mass_g"]) for o in cands)
main = cands[0]
sph_ev = op.evaluate(dict(elements=sph["elements"], back_mm=sph["back_mm"]), SPEC)

# ---------- 3-element fallback ----------
fallback = None
try:
    els3 = [dict(e, k1=0.0, a1=[0, 0, 0], k2=0.0, a2=[0, 0, 0]) for e in main["design"]["elements"]]
    els3.append(dict(r1=-60.0, r2=-90.0, t=2.5, gap=1.0, material="PMMA", k1=0.0, a1=[0, 0, 0], k2=0.0, a2=[0, 0, 0]))
    l3 = op.build_lens(els3, SPEC["epd_mm"], SPEC["half_fov_deg"], main["design"]["back_mm"])
    op.optimise(l3, SPEC["efl_mm"], weights, a.iters, aspheric_surfaces=(1, 2, 3, 4, 5, 6))
    e3, b3 = op.read_back(l3, els3)
    d3 = dict(elements=e3, back_mm=b3)
    ev3 = op.evaluate(d3, SPEC)
    if wmean(ev3["spot_um"]) < wmean(main["spot_um"]):
        fallback = dict(design=d3, spot_um=ev3["spot_um"], mass_g=ev3["mass_g"])
    print("  3-element fallback spots", [round(x, 1) for x in ev3["spot_um"]], "kept" if fallback else "DISCARDED (did not improve on 2 elements: optimiser did not converge)")
except Exception as e:  # fallback is optional
    print("  3-element fallback skipped:", repr(e)[:120])

# ---------- 2. design X-ray on the glass baseline ----------
xray = []
per_el_cost = cm.unit_cost("glass", 1)[1]
for k, e in enumerate(glass["elements"]):
    els = [x for j, x in enumerate(glass["elements"]) if j != k]
    try:
        lens = op.build_lens(els, SPEC["epd_mm"], SPEC["half_fov_deg"], glass["back_mm"])
        op.refocus(lens)
        spot = op.rms_spot_um(lens)
    except Exception:
        spot = [float("nan")] * 3
    # tolerance sensitivity of this element alone (radii only, +-0.5% 1-sigma)
    rng = np.random.default_rng(k)
    dsp = []
    for _ in range(6):
        d2 = dict(elements=[dict(x) for x in glass["elements"]], back_mm=glass["back_mm"])
        d2["elements"][k]["r1"] *= 1 + rng.normal(0, 0.005)
        d2["elements"][k]["r2"] *= 1 + rng.normal(0, 0.005)
        dsp.append(wmean(op.rms_spot_um(op.build_lens(d2["elements"], SPEC["epd_mm"], SPEC["half_fov_deg"], d2["back_mm"]))) - wmean(g_ev["spot_um"]))
    loss = wmean(spot) - wmean(g_ev["spot_um"])
    xray.append(dict(element=k + 1, material=glass["elements"][k]["material"], spot_without_um=spot, spot_increase_um=loss,
                     cost_share=per_el_cost / (per_el_cost * len(glass["elements"])), tol_sensitivity_um_per_0p5pct=float(np.mean(np.abs(dsp)))))
print("  X-ray done", f"({time.time() - t0:.0f}s)")
for x in xray:
    x["value_per_cost"] = x["spot_increase_um"] / x["cost_share"] if np.isfinite(x["spot_increase_um"]) else None
med = np.nanmedian([x["spot_increase_um"] for x in xray])
for x in xray:
    x["flag"] = "low value: candidate to drop or merge" if np.isfinite(x["spot_increase_um"]) and x["spot_increase_um"] < 0.5 * med else "keep"

# ---------- 3. tolerance Monte Carlo ----------
scales = [0.25, 0.5, 1, 2, 4]
mults = [1.5, 2, 3, 4, 6]  # quality floor = m x glass nominal RMS spot; m=2 is the pre-declared primary floor
n_mc = max(10, a.mc // 2) if a.quick else a.mc
spot_sets = {"polymer_molded": {}, "glass_ground": {}}
pool = []
for sc in scales:
    _, spots = op.monte_carlo(main["design"], SPEC, op.TOL_GRADES["polymer_molded"], n_mc, floor, seed=1, scale=sc)
    spot_sets["polymer_molded"][sc] = spots
    if sc in (0.5, 1, 2):
        pool.append(spots)
for sc in ([1, 4] if not a.quick else [1]):
    _, spots = op.monte_carlo(glass, SPEC, op.TOL_GRADES["glass_ground"], n_mc, floor, seed=2, scale=sc)
    spot_sets["glass_ground"][sc] = spots
yl = lambda spots, m: float(np.all(spots <= m * np.array(g_ev["spot_um"])[None, :], axis=1).mean())  # noqa: E731
yields = {k: {str(m): {str(sc): yl(sp, m) for sc, sp in d.items()} for m in mults} for k, d in spot_sets.items()}
print("  yield vs floor multiple (rows) at grade scale 1x:",
      {m: (round(yields["polymer_molded"][str(m)]["1"], 2), round(yields["glass_ground"][str(m)]["1"], 2)) for m in mults}, "(polymer, glass)")
m_star = next((m for m in mults if yields["polymer_molded"][str(m)]["1"] >= 0.9), None)  # smallest floor the molded lens meets
loosest = None
if m_star:
    ok = [sc for sc in scales if yields["polymer_molded"][str(m_star)][str(sc)] >= 0.9]
    loosest = max(ok) if ok else None
np.save(ROOT / "results" / "mc_spots.npy", np.concatenate(pool))

# ---------- thermal ----------
th = {}
for nm, d in (("glass", glass), ("polymer", main["design"])):
    dz = {str(T): op.thermal_defocus_mm(d, T - 25) for T in (15, 35)}
    th[nm] = dict(defocus_mm=dz, defocus_rms_um={k: op.defocus_rms_um(v, SPEC["efl_mm"] / SPEC["epd_mm"]) for k, v in dz.items()},
                  label="Estimated: literature-typical dn/dT and expansion, housing expansion ignored; verify against datasheets")

# ---------- cost / CO2 / mass with MC yield ----------
y_poly = yields["polymer_molded"][str(m_star)]["1"] if m_star else 0.02
y_gl = yields["glass_ground"]["2"]["1"]
g_cost, p_cost = cm.unit_cost("glass", 4, y_gl), cm.unit_cost("polymer", 2, y_poly)
g_co2, p_co2 = cm.co2("glass", g_ev["mass_g"]), cm.co2("polymer", main["mass_g"])
mc_cost = cm.monte_carlo_cost(2, 4, y_poly, y_gl)
rng = np.random.default_rng(0)
mounts = dict(glass=float(np.mean([cm.mount_mass_g("glass", 4, rng) for _ in range(500)])), polymer=float(np.mean([cm.mount_mass_g("polymer", 2, rng) for _ in range(500)])))
optics = dict(
    glass=dict(n_el=4, spot_um=g_ev["spot_um"], mass_g=g_ev["mass_g"], materials=sorted({e["material"] for e in glass["elements"]})),
    polymer=dict(n_el=2, spot_um=main["spot_um"], mass_g=main["mass_g"], name=main["name"], aspheric=True,
                 materials=sorted({e["material"] for e in main["design"]["elements"]}), single_material=main["single_material"]),
    polymer_sph=dict(n_el=2, spot_um=sph_ev["spot_um"], mass_g=sph_ev["mass_g"], note="spherical 2-element start, kept as an ablation"),
    polymer_3el=fallback, field_labels=op.FIELD_NAMES, field_weights=weights, quality_floor_um=floor,
    quality_floor_rule="2x the glass design's nominal RMS spot at each field (declared before polymer yields were computed)",
    mass_rel=100 * main["mass_g"] / g_ev["mass_g"],
    mass_rel_with_mount=100 * (main["mass_g"] + mounts["polymer"]) / (g_ev["mass_g"] + mounts["glass"]),
    mount_mass_g=mounts, cost_rel_polymer=[100 * v / g_cost[1] for v in p_cost], co2_rel_polymer=[100 * v / g_co2[1] for v in p_co2],
    cost_label="Estimated (placeholder coefficients; see cost_sensitivity)", cost_sensitivity=mc_cost,
    yield_mc=yields, floor_multiples=mults, required_floor_multiple_for_polymer=m_star,
    loosest_passing_polymer_grade_scale=loosest,
    floor_note=("Polymer meets a floor of %sx the glass blur at 90%% yield (pre-declared primary floor: 2x). "
                "The remaining blur must be recovered in software, which is what the restorer is for." % m_star) if m_star else
               "Polymer does not meet any floor up to 6x glass blur at 90% yield with this design.")
study = dict(floor_multiples=mults, materials=[{k: v for k, v in c.items() if k != "design"} for c in cands], xray=xray, yields=yields, thermal=th,
             floor_um=floor, tolerance_grades=op.TOL_GRADES, tol_note="process grades are ASSUMED (Estimated)",
             runtime_s=round(time.time() - t0))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "optics.json").write_text(json.dumps(optics, indent=1))
(ROOT / "results" / "design_study.json").write_text(json.dumps(study, indent=1))
DES["polymer_asph_2el"] = main["design"]
DES["candidates"] = [dict(name=c["name"], design=c["design"], weighted_spot_um=c["weighted_spot_um"]) for c in cands[:3]]
if fallback:
    DES["polymer_asph_3el"] = fallback["design"]
(ROOT / "configs" / "designs.json").write_text(json.dumps(DES, indent=1))
print(f"main polymer design: {main['name']} spots {[round(x, 1) for x in main['spot_um']]} (glass {[round(x, 1) for x in g_ev['spot_um']]}), "
      f"mass {optics['mass_rel']:.1f}% of glass, loosest passing grade x{loosest}, {time.time() - t0:.0f}s")
