"""Frozen evaluation protocol from Round 1: fixed operating point, bootstrap CIs, paired tests.

Threshold is set ONCE on the glass-rendered validation scores at 90% specificity and then
applied unchanged to every condition. The test split is touched once (Day 5).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_at_specificity(val_scores, val_labels, specificity: float = 0.90) -> float:
    neg = np.asarray(val_scores)[np.asarray(val_labels) == 0]
    return float(np.quantile(neg, specificity))


def sens_spec_at(scores, labels, thr: float) -> tuple[float, float]:
    s, y = np.asarray(scores), np.asarray(labels)
    pred = s > thr
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return sens, spec


def spec_at_sensitivity(scores, labels, sensitivity: float = 0.90) -> float:
    s, y = np.asarray(scores), np.asarray(labels)
    thr = float(np.quantile(s[y == 1], 1 - sensitivity))
    return float((s[y == 0] <= thr).mean())


def bootstrap_auc(scores, labels, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    s, y = np.asarray(scores), np.asarray(labels)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def paired_bootstrap_auc_diff(scores_a, scores_b, labels, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """AUC(a) - AUC(b) on the same resampled cases; returns (diff, lo, hi). Same images, paired.
    `level` can be raised (Bonferroni) when several pipelines are compared against glass."""
    a, b, y = map(np.asarray, (scores_a, scores_b, labels))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    q = (1 - level) / 2 * 100
    lo, hi = np.percentile(d, [q, 100 - q])
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(lo), float(hi)


def paired_bootstrap_fpr_diff(scores_a, thr_a, scores_b, thr_b, labels, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """False-positive-rate(a at thr_a) - FPR(b at thr_b) on normals, paired bootstrap."""
    a, b, y = map(np.asarray, (scores_a, scores_b, labels))
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.choice(neg, len(neg))
        d.append((a[i] > thr_a).mean() - (b[i] > thr_b).mean())
    q = (1 - level) / 2 * 100
    lo, hi = np.percentile(d, [q, 100 - q])
    return float((a[neg] > thr_a).mean() - (b[neg] > thr_b).mean()), float(lo), float(hi)


def noninferior(diff_lo: float, margin: float) -> bool:
    """Non-inferiority: the LOWER confidence bound of (candidate - reference) must stay above -margin.
    (Using the point estimate, as a plain 'drop <= 2 points' rule does, can pass by luck on a small test set.)"""
    return bool(diff_lo > -margin)


def noninferiority_verdict(auc_diff_ci, fpr_diff_ci, auc_margin: float = 0.02, fpr_margin: float = 0.02) -> dict:
    ok_auc = noninferior(auc_diff_ci[1], auc_margin)
    ok_fpr = bool(fpr_diff_ci[2] <= fpr_margin)  # upper bound of the FPR rise must stay within the margin
    return dict(auc_noninferior=ok_auc, fpr_ok=ok_fpr, pass_=ok_auc and ok_fpr)


def evaluate_condition(name, val_glass_scores, val_labels, test_scores, test_labels, thr=None, n_boot=1000):
    thr = threshold_at_specificity(val_glass_scores, val_labels) if thr is None else thr
    auc, lo, hi = bootstrap_auc(test_scores, test_labels, n_boot)
    sens, spec = sens_spec_at(test_scores, test_labels, thr)
    return dict(condition=name, auc=auc, auc_ci=(lo, hi), sens_at_fixed_thr=sens,
                fpr_on_normal=1 - spec, spec_at_90_sens=spec_at_sensitivity(test_scores, test_labels),
                n_test=int(len(test_labels)), threshold=thr)


def passes_criteria(auc_glass: float, auc_cond: float, fpr_glass: float, fpr_cond: float,
                    max_auc_drop: float = 0.02, fpr_margin: float = 0.02) -> bool:
    """Round 1 pass rule: AUC drop <= 2 points and FPR increase within margin (proposed 2 points)."""
    return (auc_glass - auc_cond) <= max_auc_drop and (fpr_cond - fpr_glass) <= fpr_margin


# ------------------------------------------------------------------ v4 additions (colleague points 16, 21) ------------
def _resample(rng, y, groups=None):
    """Case resampling; if `groups` (subject ids) is given the CLUSTER bootstrap resamples whole subjects."""
    if groups is None:
        return rng.integers(0, len(y), len(y))
    g = np.asarray(groups)
    ug = np.unique(g)
    pick = rng.choice(ug, len(ug))
    return np.concatenate([np.where(g == u)[0] for u in pick])


def cluster_auc_diff(scores_a, scores_b, labels, groups=None, n_boot=1000, seed=0, level=0.95):
    a, b, y = map(np.asarray, (scores_a, scores_b, labels))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = _resample(rng, y, groups)
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    q = (1 - level) / 2 * 100
    lo, hi = np.percentile(d, [q, 100 - q])
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(lo), float(hi)


def clinical_metrics(scores, labels, thr: float) -> dict:
    """Operating-point metrics at a threshold chosen on VALIDATION data (never on the test set)."""
    s, y = np.asarray(scores), np.asarray(labels)
    p = s > thr
    tp, fp, fn, tn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    sens = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens, specificity=tn / (tn + fp) if tn + fp else float("nan"), ppv=ppv, npv=npv, f1=f1,
                flags_everything=bool(p.all()), flags_nothing=bool((~p).all()))


def calibration(scores, labels, bins: int = 5) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    brier = float(np.mean((s - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi + (hi == 1))
        if m.any():
            ece += m.mean() * abs(s[m].mean() - y[m].mean())
            curve.append((float(s[m].mean()), float(y[m].mean()), int(m.sum())))
    return dict(brier=brier, ece=float(ece), curve=curve)


def ni_power(auc: float, n_pos: int, n_neg: int, margin: float, rho: float = 0.7, alpha: float = 0.05) -> dict:
    """Approximate power of a paired AUC non-inferiority test (true difference 0) using Hanley-McNeil variances and
    an assumed correlation rho between the two AUC estimates. Reports the n needed for 80% power."""
    from scipy.stats import norm
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)

    def var(npos, nneg):
        return (auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg)

    v = var(n_pos, n_neg)
    se = np.sqrt(max(2 * v * (1 - rho), 1e-12))
    power = float(norm.cdf(margin / se - norm.ppf(1 - alpha)))
    scale = n_pos + n_neg
    need = None
    for k in np.arange(1.0, 400.0, 0.25):
        npos2, nneg2 = int(n_pos * k), int(n_neg * k)
        if npos2 < 2 or nneg2 < 2:
            continue
        if norm.cdf(margin / np.sqrt(2 * var(npos2, nneg2) * (1 - rho)) - norm.ppf(1 - alpha)) >= 0.8:
            need = npos2 + nneg2
            break
    return dict(power=power, n_available=int(scale), n_for_80pct_power=need, underpowered=bool(power < 0.8), assumption=f"rho={rho}, true diff = 0")
