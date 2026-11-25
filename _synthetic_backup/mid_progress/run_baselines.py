"""Day-1/2 baseline: glass (4-element) vs raw polymer (2-element) in Optiland.

Usage:  python scripts/run_baselines.py            # evaluate saved designs (fast)
        python scripts/run_baselines.py --reoptimise   # re-run the radius optimisation (~2 min)
Writes results/benchmark.json and results/benchmark.md. Every row carries a provenance label.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frugaloptix import costmodel as cm  # noqa: E402
from frugaloptix.labels import Label, Metric  # noqa: E402
from frugaloptix.optics import (build_lens, element_volume_mm3, optimise_radii,  # noqa: E402
                                read_back, rms_spot_um, semi_apertures_mm)
from frugaloptix.render import render  # noqa: E402
from frugaloptix.synthetic import fundus_like  # noqa: E402

SPEC = yaml.safe_load((ROOT / "configs" / "requirement_sheet.yaml").read_text())
DESIGNS = ROOT / "configs" / "designs.json"

START = {
    "glass_4el": [dict(r1=25, r2=-70, t=4, material="N-BK7", gap=2), dict(r1=-30, r2=40, t=2.5, material="N-SF5", gap=2),
                  dict(r1=60, r2=-60, t=4, material="N-BK7", gap=2), dict(r1=40, r2=-80, t=3, material="N-BK7", gap=1)],
    "polymer_2el": [dict(r1=20, r2=-40, t=4, material="PMMA", gap=3), dict(r1=-30, r2=-60, t=3, material="Polycarbonate", gap=1)],
}


def get_designs(reopt: bool) -> dict:
    if DESIGNS.exists() and not reopt:
        return json.loads(DESIGNS.read_text())
    out = {}
    for name, els in START.items():
        lens = build_lens(els, SPEC["epd_mm"], SPEC["half_fov_deg"])
        optimise_radii(lens, SPEC["efl_mm"], max_iter=80)
        els2, back = read_back(lens, els)
        out[name] = dict(elements=els2, back_mm=back)
    DESIGNS.write_text(json.dumps(out, indent=1))
    return out


def main(reopt: bool) -> None:
    designs = get_designs(reopt)
    res, imgs = {}, {}
    ref = fundus_like(256, seed=0)
    for name, d in designs.items():
        lens = build_lens(d["elements"], SPEC["epd_mm"], SPEC["half_fov_deg"], d["back_mm"])
        spot = rms_spot_um(lens)
        semi = semi_apertures_mm(lens)[1:-1]
        vols = [element_volume_mm3(e["r1"], e["r2"], e["t"], max(semi[2 * k], semi[2 * k + 1]))
                for k, e in enumerate(d["elements"])]
        mass_g = sum(v / 1000 * cm.DENSITY_G_CM3[e["material"]] for v, e in zip(vols, d["elements"]))
        kind = "glass" if name.startswith("glass") else "polymer"
        img = render(ref, spot, px_um=SPEC["render_px_um"], noise_sigma=SPEC["noise_sigma"], seed=1)
        imgs[name] = img
        res[name] = dict(n_el=len(d["elements"]), spot=spot, mass_g=mass_g,
                         cost=cm.unit_cost(kind, len(d["elements"])), co2=cm.co2(kind, mass_g),
                         psnr=float(psnr(ref, img, data_range=1.0)), ssim=float(ssim(ref, img, data_range=1.0)),
                         efl=float(lens.paraxial.f2()) if hasattr(lens, "paraxial") else None)
    g, p = res["glass_4el"], res["polymer_2el"]
    cost_rel = {k: cm.relative(v["cost"], g["cost"]) for k, v in res.items()}
    co2_rel = {k: cm.relative(v["co2"], g["co2"]) for k, v in res.items()}
    metrics = [
        Metric("Element count (glass / polymer)", f'{g["n_el"]} / {p["n_el"]}', "", Label.SIMULATED, "counted from the design"),
        Metric("RMS spot radius, on-axis / 50% field / edge (glass)", [round(x, 1) for x in g["spot"]], "um", Label.SIMULATED, "Optiland geometric spot, 3 wavelengths"),
        Metric("RMS spot radius, on-axis / 50% field / edge (raw polymer)", [round(x, 1) for x in p["spot"]], "um", Label.SIMULATED, "same spec, same optimiser"),
        Metric("Lens mass, polymer (glass = 100)", round(100 * p["mass_g"] / g["mass_g"], 1), "rel.", Label.SIMULATED, "geometry x datasheet density"),
        Metric("Unit cost per good part, polymer low/base/high (glass base = 100)", [round(x, 1) for x in cost_rel["polymer_2el"]], "rel.", Label.ESTIMATED, "PLACEHOLDER coefficients"),
        Metric("CO2 per unit, polymer low/base/high (glass base = 100)", [round(x, 1) for x in co2_rel["polymer_2el"]], "rel.", Label.ESTIMATED, "PLACEHOLDER coefficients"),
        Metric("Cost conclusion flips across low/base/high?", cm.conclusion_flips(tuple(cost_rel["polymer_2el"])), "", Label.ESTIMATED, "Round-1 reporting rule"),
        Metric("PSNR on synthetic image: glass / raw polymer", f'{g["psnr"]:.1f} / {p["psnr"]:.1f}', "dB", Label.SIMULATED, "SYNTHETIC image, pipeline check only"),
        Metric("SSIM on synthetic image: glass / raw polymer", f'{g["ssim"]:.3f} / {p["ssim"]:.3f}', "", Label.SIMULATED, "SYNTHETIC image, pipeline check only"),
        Metric("Disease-detection AUC (A/B/C/D)", "not run", "", Label.TARGET, "needs RFMiD/IDRiD + restorer (Day 3-5)"),
    ]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "optics.json").write_text(json.dumps(dict(
        glass=dict(n_el=g["n_el"], spot_um=g["spot"], mass_g=g["mass_g"]),
        polymer=dict(n_el=p["n_el"], spot_um=p["spot"], mass_g=p["mass_g"]),
        mass_rel=100 * p["mass_g"] / g["mass_g"], cost_rel_polymer=cost_rel["polymer_2el"],
        co2_rel_polymer=co2_rel["polymer_2el"], cost_label="Estimated (placeholder coefficients)",
        field_labels=["on-axis", "50% field", "edge"]), indent=1))
    (out / "benchmark.json").write_text(json.dumps([m.to_dict() for m in metrics], indent=1, default=str))
    md = ["| Metric | Value | Unit | Label | Note |", "|---|---|---|---|---|"]
    md += [f"| {m.name} | {m.value} | {m.unit} | **{m.label.value}** | {m.note} |" for m in metrics]
    (out / "benchmark.md").write_text("\n".join(md) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
        for a, (t, im) in zip(ax, [("Original (synthetic)", ref), ("Glass-rendered", imgs["glass_4el"]),
                                   ("Raw polymer-rendered", imgs["polymer_2el"])]):
            a.imshow(im, cmap="gray", vmin=0, vmax=1)
            a.set_title(t, fontsize=9)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / "render_comparison.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print((out / "benchmark.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reoptimise", action="store_true")
    main(ap.parse_args().reoptimise)
