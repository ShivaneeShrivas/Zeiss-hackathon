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


def paired_bootstrap_auc_diff(scores_a, scores_b, labels, n_boot: int = 1000, seed: int = 0):
    """AUC(a) - AUC(b) on the same resampled cases; returns (diff, lo, hi). Same images, paired."""
    a, b, y = map(np.asarray, (scores_a, scores_b, labels))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(lo), float(hi)


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
