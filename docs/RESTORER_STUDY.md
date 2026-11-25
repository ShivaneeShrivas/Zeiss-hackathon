# Restorer study (real RFMiD run, validation split only)

Four restorer variants were trained for 20 epochs and compared on the **validation** split. The test split was not used to choose.
Raw plastic lens on validation: PSNR 32.7 dB, AUC 0.963 through the frozen glass-trained classifier.

| Variant | PSNR gain vs raw plastic | Validation AUC (frozen classifier) | Note |
|---|---|---|---|
| Fidelity only (`fid_s0`) | +10.6 dB | 0.967 | sharpest images |
| Lesion-weighted (`lesion_s0`) | +8.8 dB | 0.960 | |
| Lesion-weighted + classifier loss (`lesion_task_s0`) | +7.5 dB | **0.975** | **selected** (highest validation AUC) |
| Lesion-weighted, glass target (`lesion_glass_s0`) | +4.9 dB | 0.967 | |

Selection rule (from `configs/acceptance.yaml`): highest validation AUC through the frozen glass-trained classifier, then lesion recall, then PSNR gain.

**Caveat.** The selected variant was trained with a loss from the same frozen classifier that scores it. On the test set an independent MobileNet sees no AUC gain from restoration (−0.003, 95% CI −0.028 to +0.023). RFMiD has no lesion masks, so lesion recall and invented-lesion checks were not run. **No variant met the restoration-claim criteria, so restoration is not claimed.**

Full numbers: `results/restorer_study.json`, `results/results.json`.
