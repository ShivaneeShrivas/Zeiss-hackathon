# Running FrugalOptiX v2 (all loopholes addressed in code)

    pip install -r requirements.txt
    bash scripts/run_all.sh synthetic            # ~15 min on 2 CPUs, smoke test only (NOT scientific)
    bash scripts/run_all.sh rfmid data/RFMiD     # real run (use a GPU / Colab, add --pretrained)
    SEEDS="0 1 2" bash scripts/run_all.sh rfmid data/RFMiD --pretrained
    python -m http.server 8000 --directory web   # demo; phone timing: /bench.html

Order: design study -> render A,B -> classifiers B (frozen) and A (ceiling) -> Grad-CAM saliency -> field-weighted design ->
render C (nominal) and X (independent renderer) -> baseline E + restorer (+ fidelity-only ablation) -> ONE evaluation -> ONNX.
Then `python scripts/04_evaluate.py --out results/seedN.json` per seed and `python scripts/06_aggregate.py results/seed*.json`.
Real optics: docs/REAL_CAPTURE.md + scripts/08_real_capture_eval.py. 19 unit tests: `pytest -q`.

What a synthetic run shows: the machinery works. With 36 test images every verdict is "not shown"; that is expected, not a result.
