# Clinical and regulatory roadmap

**Status: research prototype. Not a medical device. Not for diagnosis or patient care.**

## Intended use (for the future product, not today)
A low-cost handheld retinal camera that helps a trained operator capture a gradable fundus image and flags images for review by a clinician. It never gives a diagnosis.

## Stages
| Stage | What | Needs |
|---|---|---|
| 0 (now) | Simulation and public-data study; claims labelled Measured / Simulated / Estimated / Target | this repo |
| 1 | Bench: slanted-edge MTF and model-eye images through a real polymer lens (docs/MODEL_EYE_BENCH.md) | lens samples, model eye, sensor |
| 2 | Retrospective study on IDRiD / DDR / RFMiD and a local dataset, with ophthalmologist-graded lesions | ethics approval, data agreement |
| 3 | Prospective pilot, same patients imaged with reference camera and prototype | IRB/IEC, clinical partner |
| 4 | Quality system and regulatory file | ISO 13485 QMS, ISO 14971 risk file, IEC 62304 software, IEC 60601-1 and ISO 15004-2 / ISO 10940 (light safety, fundus camera), CDSCO medical device registration (India) |

## Risks to manage (ISO 14971 starting list)
- AI hides or invents a lesion (mitigation: show original beside restored image; lesion-level tests; uncertainty gate).
- Poor capture trusted as normal (mitigation: separate quality gate with recapture advice).
- Light exposure to the retina (mitigation: illumination safety analysis before any human use).
- Distribution shift (new camera, ethnicity, disease mix) (mitigation: external validation, drift monitoring).

## Honest limits today
Synthetic images only in this repo's results; no human imaging; no clinician review; cost and carbon inputs are assumptions.
