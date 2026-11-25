"""Wide-field montage from overlapping narrow-field captures (colleague point 3, route 2): registration by phase correlation,
feathered blending, and quality checks (overlap, residual, coverage). SIMULATED test on synthetic fundus tiles."""
from __future__ import annotations

import numpy as np
from skimage.filters import sobel, window
from skimage.registration import phase_cross_correlation


def make_tiles(img: np.ndarray, tile: int, step: int, jitter: float, rng) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Cut overlapping tiles from a large image at a grid of nominal positions, each with an UNKNOWN integer jitter (hand-held aiming error)."""
    H, W = img.shape[:2]
    tiles, pos = [], []
    for y in range(0, H - tile + 1, step):
        for x in range(0, W - tile + 1, step):
            jy, jx = (int(round(rng.uniform(-jitter, jitter))) for _ in range(2))
            yy, xx = int(np.clip(y + jy, 0, H - tile)), int(np.clip(x + jx, 0, W - tile))
            tiles.append(img[yy:yy + tile, xx:xx + tile].copy())
            pos.append((yy, xx))
    return tiles, pos


def stitch(tiles, nominal, tile: int, canvas_hw: tuple[int, int], search: int = 10):
    """Register every tile to the growing mosaic and blend. Returns mosaic, estimated positions, per-tile checks."""
    H, W = canvas_hw
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    ramp = np.minimum.outer(np.minimum(np.arange(tile), np.arange(tile)[::-1]) + 1.0, np.minimum(np.arange(tile), np.arange(tile)[::-1]) + 1.0)
    est, checks = [], []
    for k, (t, (ny, nx)) in enumerate(zip(tiles, nominal)):
        g = t.astype(np.float64)
        if k == 0:
            py, px, resid, ov = ny, nx, 0.0, 1.0
        else:
            mos = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0)
            y0, x0 = int(np.clip(ny, 0, H - tile)), int(np.clip(nx, 0, W - tile))
            region = mos[y0:y0 + tile, x0:x0 + tile]
            m = (wsum[y0:y0 + tile, x0:x0 + tile] > 0)
            ov = float(m.mean())
            if ov > 0.15:
                best, py, px = -2.0, y0, x0
                for dy in range(-search, search + 1):          # exhaustive normalised cross-correlation inside the aiming-error window
                    for dx in range(-search, search + 1):
                        yy, xx = y0 + dy, x0 + dx
                        if yy < 0 or xx < 0 or yy + tile > H or xx + tile > W:
                            continue
                        mm = wsum[yy:yy + tile, xx:xx + tile] > 0
                        if mm.sum() < 0.1 * tile * tile:
                            continue
                        a_, b_ = mos[yy:yy + tile, xx:xx + tile][mm], g[mm]
                        a_, b_ = a_ - a_.mean(), b_ - b_.mean()
                        ncc = float((a_ * b_).sum() / (np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum()) + 1e-9))
                        if ncc > best:
                            best, py, px = ncc, yy, xx
                resid = 1.0 - best                                  # 0 = perfect overlap match; high = do not trust this tile
            else:
                py, px, resid = y0, x0, float("nan")
        acc[py:py + tile, px:px + tile] += g * ramp
        wsum[py:py + tile, px:px + tile] += ramp
        est.append((py, px))
        checks.append(dict(overlap=ov, residual=resid, ok=bool(ov > 0.15 or k == 0)))
    mosaic = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0)
    return mosaic, est, dict(coverage=float((wsum > 0).mean()), tiles=checks, missing_regions=float((wsum == 0).mean()))
