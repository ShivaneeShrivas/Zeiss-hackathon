"""ONE geometric transform for an image and ALL of its masks (fixes the crop-vs-resize misalignment).

The old pipeline centre-cropped images to a square (dropping ~1/3 of a 4288x2848 IDRiD photo) but resized masks
whole, so lesion masks no longer lined up with the pixels. Here the full field of view is kept: the image is
scaled so its long side equals `size` and padded (black, like the fundus background) to a square. Images use bilinear
resampling, masks nearest-neighbour, with identical scale and offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Letterbox:
    src_w: int
    src_h: int
    size: int
    scale: float
    new_w: int
    new_h: int
    off_x: int
    off_y: int

    def point(self, x: float, y: float) -> tuple[float, float]:
        """Map a source-pixel coordinate (centre-of-pixel convention) to the output square."""
        return ((x + 0.5) * self.scale - 0.5 + self.off_x, (y + 0.5) * self.scale - 0.5 + self.off_y)

    def valid_mask(self) -> np.ndarray:
        m = np.zeros((self.size, self.size), bool)
        m[self.off_y:self.off_y + self.new_h, self.off_x:self.off_x + self.new_w] = True
        return m


def plan(w: int, h: int, size: int) -> Letterbox:
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return Letterbox(w, h, size, s, nw, nh, (size - nw) // 2, (size - nh) // 2)


def apply(im: Image.Image, lb: Letterbox, resample, fill=0) -> np.ndarray:
    if im.size != (lb.src_w, lb.src_h):
        raise ValueError(f"size mismatch: image/mask is {im.size}, expected {(lb.src_w, lb.src_h)}")
    r = im.resize((lb.new_w, lb.new_h), resample)
    canvas = Image.new(im.mode, (lb.size, lb.size), fill)
    canvas.paste(r, (lb.off_x, lb.off_y))
    return np.asarray(canvas)


def load_pair(image_path, size: int, mask_paths: dict[str, Path] | None = None):
    """-> (uint8 HxWx3 image, {class: bool HxW mask}, Letterbox). Raises if any mask size differs from the image."""
    im = Image.open(image_path).convert("RGB")
    lb = plan(*im.size, size)
    out = {}
    for name, p in (mask_paths or {}).items():
        if p is None or not Path(p).exists():
            out[name] = np.zeros((size, size), bool)
            continue
        out[name] = apply(Image.open(p).convert("L"), lb, Image.NEAREST) > 127
    return apply(im, lb, Image.BILINEAR), out, lb


def find_masks(masks_dir, image_id: str, classes=None) -> dict[str, Path | None]:
    """Layouts: <dir>/<id>.png (single class 'lesion')  or  <dir>/<CLASS>/<id>.(png|tif|jpg) (IDRiD-style)."""
    d = Path(masks_dir)
    if classes is None:
        classes = sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []
    if not classes:
        p = d / f"{image_id}.png"
        return {"lesion": p if p.exists() else None}
    found = {}
    for c in classes:
        cand = [q for ext in ("png", "tif", "tiff", "jpg") for q in [d / c / f"{image_id}.{ext}", d / c / f"{image_id}_{c}.{ext}"] if q.exists()]
        found[c] = cand[0] if cand else None
    return found
