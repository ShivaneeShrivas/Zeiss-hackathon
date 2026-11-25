"""Cost per FUNCTIONAL (quality-approved) camera module, complete-module mass/power, and cradle-to-grave CO2e with ranges.
Inputs: configs/cost_inputs.yaml (every number [low, likely, high] + status). Optical yields come from results/optics_extended.json.
Monte Carlo (triangular) -> median and 90% interval; P(polymer cheaper); break-even tooling cost and yield.  Estimated, not quoted."""
import json

import numpy as np
import yaml

from _common import ROOT

N = 20000
rng = np.random.default_rng(0)
IN = yaml.safe_load((ROOT / "configs" / "cost_inputs.yaml").read_text())
OPT = json.loads((ROOT / "results" / "optics.json").read_text())
EXT = json.loads((ROOT / "results" / "optics_extended.json").read_text()) if (ROOT / "results" / "optics_extended.json").exists() else {}
mass_el = {"glass": OPT.get("glass", {}).get("mass_g", 10.0), "polymer": OPT.get("polymer", {}).get("mass_g", 1.5)}


def tri(lmh, n=N):
    lo, mo, hi = lmh
    return rng.triangular(lo, mo, hi, n) if hi > lo else np.full(n, float(lo))


def v(sec, key):
    return tri(sec[key]["v"] if "v" in sec[key] else sec[key])


def pct(x):
    return dict(p5=float(np.percentile(x, 5)), median=float(np.median(x)), p95=float(np.percentile(x, 95)))


tolr = EXT.get("tolerance", {}).get("runs", {})
yg = tolr.get("glass_ground", {}).get("robustness_yield_125pct_of_own_nominal", {}).get("yield_", 0.97)
yp = {k: tolr.get(k, {}).get("robustness_yield_125pct_of_own_nominal", {}).get("yield_") for k in ("polymer_commercial", "polymer_precision")}
G, P, C = IN["glass"], IN["polymer"], IN["common"]
out = dict(label="Estimated: ranges from configs/cost_inputs.yaml (status per row). Replace 'assumed' rows with quotations.", scenarios={})
common_cost = sum(tri(C[k]["v"]) for k in ("sensor", "illumination_led_driver", "pcb_electronics", "focus_mechanism", "body_housing", "packaging_logistics"))
for vol in IN["volumes_modules"]:
    res = {}
    for grade in ("polymer_commercial", "polymer_precision"):
        y_opt = yp[grade] if yp[grade] is not None else 0.8
        # ---- glass module ----
        scrap = tri(G["element_scrap_rate"]["v"])
        g_lens = G["n_elements"] * (tri(G["element_cost"]["v"]) / (1 - scrap) + tri(G["coating_per_element"]["v"])) + tri(G["barrel_per_module"]["v"]) \
            + tri(G["assembly_per_module"]["v"]) + tri(G["inspection_per_module"]["v"]) + tri(G["tooling_total"]["v"]) / vol
        g_good = g_lens / yg
        # ---- polymer module ----
        cav = tri(P["cavities"]["v"])
        shots_cost = tri(P["cycle_time_s"]["v"]) / 3600 * tri(P["machine_rate_per_hour"]["v"]) / cav
        mat_cost = mass_el["polymer"] / P["n_elements"] / 1000 * tri(P["material_per_kg"]["v"])
        tool = tri(P["tooling_per_element"]["v"]) * P["n_elements"] / np.minimum(vol, tri(P["mould_life_shots"]["v"]) * cav)   # extra tools once the mould life is exhausted
        p_lens = P["n_elements"] * (shots_cost + mat_cost + tri(P["coating_per_element"]["v"])) + tool + tri(P["assembly_per_module"]["v"]) + tri(P["inspection_per_module"]["v"]) + tri(P["housing_per_module"]["v"])
        p_good = p_lens / y_opt
        repl = tri(C["replacement_rate_over_life"]["v"])
        tot_g, tot_p = (g_good + common_cost) * (1 + repl), (p_good + common_cost) * (1 + repl)
        res[grade] = dict(optical_yield_used=float(y_opt), glass_lens_module=pct(g_good), polymer_lens_module=pct(p_good), glass_camera_module=pct(tot_g), polymer_camera_module=pct(tot_p),
                          saving_camera_module=pct(tot_g - tot_p), p_polymer_cheaper_camera_module=float(np.mean(tot_p < tot_g)),
                          p_polymer_cheaper_lens_module=float(np.mean(p_good < g_good)), optical_share_of_camera_cost_glass=float(np.median(g_good / tot_g)))
    out["scenarios"][str(vol)] = res
# break-even: optical-module yield at which polymer lens module stops being cheaper (median inputs, 10k volume)
g_med = float(np.median(g_good)); p_raw = float(np.median(p_lens))
out["break_even"] = dict(polymer_yield_needed_for_lens_parity_at_last_volume=float(min(1.0, p_raw / g_med)), note="at the largest volume with median inputs; values near 1 mean polymer needs a near-perfect process to match glass")
# ---------------- complete-module mass & power ----------------
M = IN["mass_g_common"]
mc = tri(M["sensor_pcb_led"]) + tri(M["focus_mechanism"]) + tri(M["body"])
out["module_mass_g"] = dict(common=pct(mc), glass_total=pct(mc + mass_el["glass"] + tri([2, 4, 8])), polymer_total=pct(mc + mass_el["polymer"] + tri([0.5, 1, 2])),
                            optics_only_saving_pct=float(100 * (1 - mass_el["polymer"] / mass_el["glass"])),
                            module_saving_pct=pct(100 * (1 - (mc + mass_el["polymer"] + 1) / (mc + mass_el["glass"] + 4))))
out["compute_options"] = {k: dict(incremental_cost=pct(tri(v["incremental_cost"])), power_w=pct(tri(v["power_w"])), mass_g=pct(tri(v["mass_g"]))) for k, v in C["compute_options"].items()}
# ---------------- LCA (cradle-to-grave, per camera-module optics + mount) ----------------
L = IN["lca"]
grid = tri(L["grid_kgco2_per_kwh"]["v"])
mat_pol = {"PMMA": "pmma_kgco2_per_kg", "Polycarbonate": "pc_kgco2_per_kg", "Zeonex": "cop_coc_kgco2_per_kg"}
pol_mats = OPT.get("polymer", {}).get("materials", ["PMMA", "Polycarbonate"])
fac = np.mean([tri(L[mat_pol.get(m, "pmma_kgco2_per_kg")]["v"]) for m in pol_mats], axis=0)
def life(kind, vol, mixed=True):
    if kind == "glass":
        kg = mass_el["glass"] / 1000
        raw = kg * tri(L["glass_kgco2_per_kg"]["v"]) * 1.5            # x1.5 for blank-to-element yield loss (assumed)
        proc = G["n_elements"] * tri(L["glass_grind_polish_kwh_per_element"]["v"]) * grid
        tool = tri(L["tooling_kgco2_per_tool"]["v"]) / vol
        mount = tri(L["barrel_kgco2"]["glass_metal_barrel"])
        eol = tri(L["end_of_life_kgco2_per_module"]["glass"])
    else:
        kg = mass_el["polymer"] / 1000
        raw = kg * fac * 1.1
        proc = kg * tri(L["moulding_kwh_per_kg"]["v"]) * grid
        tool = P["n_elements"] * tri(L["tooling_kgco2_per_tool"]["v"]) / vol
        mount = tri(L["barrel_kgco2"]["polymer_mount"])
        eol = tri(L["end_of_life_kgco2_per_module"]["polymer_mixed" if mixed else "polymer_single_material"])
    return raw + proc + tool + mount + tri(L["transport_kgco2_per_module"]["v"]) + eol
out["lca"] = {}
for vol in IN["volumes_modules"]:
    lg, lp, ls = life("glass", vol), life("polymer", vol, True), life("polymer", vol, False)
    out["lca"][str(vol)] = dict(glass=pct(lg), polymer_mixed=pct(lp), polymer_single_material=pct(ls), saving_pct=pct(100 * (1 - lp / lg)),
                                p_polymer_lower=float(np.mean(lp < lg)),
                                note="Per-module cradle-to-grave of optics+mount only; camera electronics, sensor and housing are common to both and excluded. Optical yield losses scale both.")
out["lca_boundary"] = "raw material, processing energy, tooling, mount, transport, end-of-life; excludes use phase (none) and shared electronics. Estimated; factors in cost_inputs.yaml"
(ROOT / "results" / "cost_model.json").write_text(json.dumps(out, indent=1))
s = out["scenarios"][str(IN["volumes_modules"][1])]
print("10k modules, camera-module saving (glass-polymer, USD):", {k: round(v["saving_camera_module"]["median"], 1) for k, v in s.items()}, "P(polymer cheaper):", {k: round(v["p_polymer_cheaper_camera_module"], 2) for k, v in s.items()})
print("module mass saving % (median):", round(out["module_mass_g"]["module_saving_pct"]["median"], 1), "| optics-only:", round(out["module_mass_g"]["optics_only_saving_pct"], 1))
print("LCA 10k, polymer vs glass saving % median:", round(out["lca"][str(IN["volumes_modules"][1])]["saving_pct"]["median"], 1))
