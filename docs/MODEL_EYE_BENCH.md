# Physical model-eye experiment

Goal: replace simulated degradation with measured degradation.

1. **Targets**: a slanted-edge chart (ISO 12233) and a retinal phantom / printed fundus transparency at the model-eye retina plane.
2. **Cameras**: (a) reference glass lens, (b) polymer lens. Same sensor, same illumination, same exposure. Fix focus on the centre, then record at 0, 0.5 and 1.0 of the field.
3. **MTF**: run `python scripts/12_slanted_edge_mtf.py --image edge.png --px-um <pixel pitch>` on each field. Compare with `results/optics_extended.json` (simulated MTF). Report the gap; large gaps mean the simulation or tolerances are wrong.
4. **PSF capture**: image a pinhole or point source at each field to build a measured PSF bank. Put it in `results/psf_bank.npz` (same shape) and re-run `01_prepare_renders.py --conds P`.
5. **Paired images**: image the same phantom with both lenses, register (scripts/11_montage_demo.py has the registration code) and evaluate the restorer on real polymer images.
6. **Record**: lens serial, temperature, date, exposure. Never mix lens samples in one number.

Pass rule: use the same thresholds as `configs/acceptance.yaml`, declared before measuring.
