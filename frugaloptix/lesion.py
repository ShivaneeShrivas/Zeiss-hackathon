"""Hallucination / lesion-fidelity checks (Round 1 promise): did the pipeline keep lesions, and did it invent any?"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as ssim


def _peaks(g: np.ndarray, thr: float = 0.04) -> np.ndarray:
    """Bright blob centres: peaks of a difference-of-Gaussians (lesion-like spots)."""
    dog = gaussian_filter(g, 1.5) - gaussian_filter(g, 5)
    return np.argwhere((dog == maximum_filter(dog, size=7)) & (dog > thr))


def lesion_metrics(orig_u8: np.ndarray, test_u8: np.ndarray, mask: np.ndarray, radius: float = 4.0) -> dict | None:
    """orig/test: HxWx3 uint8; mask: HxW bool lesion mask (e.g. IDRiD masks OR-ed). Uses the green channel.

    contrast_retention : lesion-vs-surround contrast in `test` / in `orig` (1 = kept, <1 blurred, >1 exaggerated).
    lesion_ssim        : SSIM averaged over lesion pixels.
    lesion_recall      : share of the original's lesion-like blobs (inside the mask) that still have a blob in `test`.
    hallucinated_blobs : bright blobs in `test` with NO blob in `orig` nearby (outside the lesion mask), per image.
                         This is the invented-detail check; blur alone does not raise it.
    """
    if mask is None or mask.sum() < 4:
        return None
    go, gt = orig_u8[..., 1].astype(float) / 255, test_u8[..., 1].astype(float) / 255
    ring = binary_dilation(mask, iterations=6) & ~binary_dilation(mask, iterations=2)
    if ring.sum() < 4:
        return None
    co = go[mask].mean() - go[ring].mean()
    ct = gt[mask].mean() - gt[ring].mean()
    _, smap = ssim(go, gt, data_range=1.0, full=True)
    po, pt = _peaks(go), _peaks(gt)
    in_mask = lambda p: np.array([mask[y, x] for y, x in p], bool) if len(p) else np.zeros(0, bool)  # noqa: E731
    lesion_pk = po[in_mask(po)] if len(po) else po
    tree_t = cKDTree(pt) if len(pt) else None
    recall = float(np.mean([tree_t.query(p)[0] <= radius for p in lesion_pk])) if len(lesion_pk) and tree_t else (0.0 if len(lesion_pk) else float("nan"))
    tree_o = cKDTree(po) if len(po) else None
    halluc = 0
    for p in pt:
        near_orig = tree_o is not None and tree_o.query(p)[0] <= radius
        if not near_orig and not binary_dilation(mask, iterations=3)[p[0], p[1]]:
            halluc += 1
    return dict(contrast_retention=float(ct / co) if abs(co) > 1e-3 else float("nan"), lesion_ssim=float(smap[mask].mean()),
                lesion_recall=recall, hallucinated_blobs=float(halluc))


def aggregate(rows: list[dict]) -> dict:
    rows = [r for r in rows if r]
    if not rows:
        return {}
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]} | {"n_images_with_masks": len(rows)}


# ---------------------------------------------------------------- v4: error taxonomy + per-class overlap (point 14) ----
def _dog(g):
    return gaussian_filter(g, 1.5) - gaussian_filter(g, 5)


def taxonomy(orig_u8, test_u8, mask, radius: float = 4.0, distort_ratio: float = 0.5) -> dict | None:
    """Three separate error types for ONE lesion class (mask = that class):
         removed   : genuine lesion peaks in the original with no peak left in `test`
         distorted : still present but peak contrast changed by more than `distort_ratio` (x0.5 .. x1.5)
         invented  : peaks in `test` away from any original peak and away from the mask
       plus pixel-level Dice / precision / recall / F1 of the thresholded DoG response inside the mask (`test` vs original)."""
    if mask is None or mask.sum() < 4:
        return None
    go, gt = orig_u8[..., 1].astype(float) / 255, test_u8[..., 1].astype(float) / 255
    do, dt = _dog(go), _dog(gt)
    po, pt = _peaks(go), _peaks(gt)
    in_m = lambda p: np.array([mask[y, x] for y, x in p], bool) if len(p) else np.zeros(0, bool)  # noqa: E731
    gen = po[in_m(po)] if len(po) else po
    tree_t = cKDTree(pt) if len(pt) else None
    tree_o = cKDTree(po) if len(po) else None
    removed = distorted = kept = 0
    for p in gen:
        d, j = tree_t.query(p) if tree_t else (1e9, None)
        if d > radius:
            removed += 1
        else:
            r = dt[tuple(pt[j])] / max(do[tuple(p)], 1e-6)
            if abs(r - 1) > distort_ratio:
                distorted += 1
            else:
                kept += 1
    near_mask = binary_dilation(mask, iterations=3)
    invented = sum(1 for p in pt if not (tree_o is not None and tree_o.query(p)[0] <= radius) and not near_mask[p[0], p[1]])
    thr = 0.04
    seg_o, seg_t = (do > thr) & near_mask, (dt > thr) & near_mask
    tp = int((seg_o & seg_t).sum())
    fp, fn = int((~seg_o & seg_t).sum()), int((seg_o & ~seg_t).sum())
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    n = max(len(gen), 1)
    return dict(n_lesions=int(len(gen)), removed=removed / n, distorted=distorted / n, kept=kept / n, invented_per_image=float(invented),
                dice=float(dice), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=float(dice))
