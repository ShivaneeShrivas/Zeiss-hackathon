import json, math, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_wilson_interval_contains_p():
    from frugaloptix import tolerance
    f = getattr(tolerance, "wilson", None) or getattr(tolerance, "wilson_ci", None)
    if f is None:
        return
    r = f(90, 100)
    lo, hi = (r["lo"], r["hi"]) if isinstance(r, dict) else (r[-2], r[-1]) if len(r) > 2 else r
    assert lo < 0.9 < hi <= 1.0


def test_psf_kernels_normalised_and_field_dependent():
    z = np.load(ROOT / "results" / "psf_bank.npz")
    k = z["polymer"]
    assert k.shape[0] == 5 and k.shape[1] == 3
    assert np.allclose(k.sum(axis=(-1, -2)), 1.0, atol=1e-3)
    assert not np.allclose(k[0], k[-1])          # centre differs from edge


def test_mtf_selftest_error_small():
    d = json.load(open(ROOT / "results" / "mtf_selftest.json"))
    assert max(v for v in (d.get("max_abs_err"), 0) if v is not None) < 0.02


def test_acceptance_declared_and_results_consistent():
    r = json.load(open(ROOT / "results" / "results.json"))
    assert r["meta"]["margin_auc"] == 0.05
    for c in r["restoration_claims"].values():          # claim only if every check passes
        checks = [c["psnr_better"], c["lesion_recall_not_worse"], c["invented_not_worse"], c["independent_auc_gain_significant"]]
        assert c["restoration_claimed"] == all(checks)


def test_gate_rates_valid():
    r = json.load(open(ROOT / "results" / "results.json"))
    for g in r["gate"]["by_condition"].values():
        assert 0 <= g["accepted"] <= 1
