"""PSF thumbnails for the demo (glass row, polymer row, 5 field positions; green channel, shared intensity scale per column)."""
import numpy as np
from PIL import Image

from _common import ROOT

z = np.load(ROOT / "results" / "psf_bank.npz")
cells = []
for name in ("glass", "polymer"):
    k = z[name][:, 1]                      # [field, 41, 41] green
    cells.append(k)
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402
rows = []
for k in cells:
    row = []
    for f in range(k.shape[0]):
        c = k[f][10:31, 10:31]
        c = c / max(k[f].max(), 1e-12)
        row.append((cm.magma(c ** 0.6)[..., :3] * 255).astype(np.uint8))
    rows.append(np.concatenate(row, axis=1))
img = np.concatenate(rows, axis=0)
Image.fromarray(img).resize((img.shape[1] * 6, img.shape[0] * 6), Image.NEAREST).save(ROOT / "web" / "assets" / "psf_grid.png")
print("saved", img.shape)
