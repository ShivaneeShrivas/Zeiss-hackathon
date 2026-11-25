"""Seeded splits and SHA-256 checksum manifest so every run is reproducible and auditable."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_split(ids, labels, frac=(0.7, 0.15, 0.15), seed: int = 1234) -> dict[str, list]:
    """Deterministic stratified train/val/test split on (id, binary label)."""
    ids, labels = np.asarray(ids), np.asarray(labels)
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for cls in np.unique(labels):
        sel = ids[labels == cls]
        sel = sel[rng.permutation(len(sel))]
        n_tr, n_va = int(round(frac[0] * len(sel))), int(round(frac[1] * len(sel)))
        out["train"] += sel[:n_tr].tolist()
        out["val"] += sel[n_tr:n_tr + n_va].tolist()
        out["test"] += sel[n_tr + n_va:].tolist()
    return {k: sorted(v) for k, v in out.items()}


def write_manifest(image_dir: str | Path, out_json: str | Path, splits: dict | None = None) -> dict:
    image_dir = Path(image_dir)
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"})
    manifest = {"n_files": len(files), "files": {str(p.relative_to(image_dir)): sha256(p) for p in files}}
    if splits:
        manifest["splits"] = splits
    Path(out_json).write_text(json.dumps(manifest, indent=1))
    return manifest
