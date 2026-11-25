# Real-data run: RFMiD (Kaggle, GPU, 1 seed, 20 restorer epochs)

Test set: 480 held-out images (360 abnormal, 120 normal), 256 px. Lens blur is simulated (ray-traced PSF, Gaussian, independent renderer). No physical lens, no lesion masks.

| Pipeline (nominal blur) | AUC | vs glass (95% CI) | Verdict |
|---|---|---|---|
| Glass lens | 0.980 | - | reference |
| Plastic, raw | 0.949 | -0.031 (-0.054, -0.012) | not non-inferior |
| Plastic + AI restorer | 0.970 | -0.010 (-0.022, 0.000) | AUC within margin, false-positive rule not met |
| Plastic, classifier retrained | 0.952 | -0.028 (-0.051, -0.009) | not non-inferior |

Caveats stated in the demo:
- The restorer variant chosen on validation was trained with a classifier loss against the same frozen classifier used to score it. An independent MobileNet shows no AUC gain from the restorer (-0.003, CI -0.028 to 0.023), so the 0.970 is not independent proof.
- RFMiD has no lesion masks, so lesion recall / invented lesions are not assessed and restoration is not claimed.
- The quality gate rejects 100% of the independent-renderer images (limits fitted on nominal images).
- Polymer lens misses the predeclared MTF thresholds (worst ratio 0.19 of glass).
- Dataset credit: Pachade et al., "Retinal Fundus Multi-Disease Image Dataset (RFMiD)", Data 2021. Check the licence before redistributing images.
