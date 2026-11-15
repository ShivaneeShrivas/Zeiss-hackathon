import numpy as np
from skimage.metrics import structural_similarity as ssim

from frugaloptix import costmodel as cm
from frugaloptix.evaluation import (bootstrap_auc, paired_bootstrap_auc_diff, passes_criteria,
                                    sens_spec_at, threshold_at_specificity)
from frugaloptix.render import render
from frugaloptix.splits import stratified_split
from frugaloptix.synthetic import fundus_like


def _scores(n=600, shift=1.2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y + 0, rng.normal(0, 1, n) + shift * y


def test_threshold_gives_requested_specificity():
    y, s = _scores(4000)
    thr = threshold_at_specificity(s, y, 0.90)
    _, spec = sens_spec_at(s, y, thr)
    assert abs(spec - 0.90) < 0.01


def test_bootstrap_ci_contains_point_estimate_and_paired_zero_for_identical():
    y, s = _scores()
    auc, lo, hi = bootstrap_auc(s, y, 300)
    assert lo <= auc <= hi
    d, dlo, dhi = paired_bootstrap_auc_diff(s, s, y, 100)
    assert d == 0 and dlo == 0 and dhi == 0


def test_pass_rule():
    assert passes_criteria(0.90, 0.885, 0.10, 0.11)
    assert not passes_criteria(0.90, 0.85, 0.10, 0.10)


def test_split_is_deterministic_disjoint_and_stratified():
    ids = np.arange(200)
    lab = (ids % 5 == 0).astype(int)
    a, b = stratified_split(ids, lab, seed=7), stratified_split(ids, lab, seed=7)
    assert a == b
    allids = a["train"] + a["val"] + a["test"]
    assert sorted(allids) == list(range(200))
    assert len(set(allids)) == 200


def test_render_blurs_more_at_edge_when_spot_grows():
    img = fundus_like(128)
    out = render(img, [1.0, 20.0, 60.0], px_um=10.0)
    c = (slice(54, 74), slice(54, 74))
    e = (slice(0, 20), slice(54, 74))
    assert ssim(img[c], out[c], data_range=1.0, win_size=7) > ssim(img[e], out[e], data_range=1.0, win_size=7)


def test_cost_model_and_flip_rule():
    g = cm.unit_cost("glass", 4)
    assert cm.relative(g, g)[1] == 100.0
    assert cm.conclusion_flips((90, 95, 110)) and not cm.conclusion_flips((50, 60, 70))
