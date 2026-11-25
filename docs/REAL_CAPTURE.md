# Real-capture protocol (the one thing a simulation cannot replace)
1. Print/mold or buy a 2-element PMMA/PC lens matching `configs/designs.json` (or any cheap lens as a stand-in; say which).
2. Display 100+ test images (50 normal, 50 abnormal, from the TEST split only) on a calibrated screen at fixed distance, or use printed fundus charts / a model eye.
3. Photograph each through the lens with a phone, fixed exposure. Save as `capture/normal/*.png`, `capture/abnormal/*.png`.
4. `python scripts/08_real_capture_eval.py --dir capture`  -> AUC raw vs restored, labelled **Measured (real optics)**.
5. Fine-tune the restorer on half the captures and test on the other half to close the sim-to-real gap; report both numbers.
Limits to state: screen-photo is a proxy for a retina, not clinical evidence.
