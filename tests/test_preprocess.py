import numpy as np
from PIL import Image

from frugaloptix.preprocess import apply, load_pair, plan


def _scene(tmp_path, w=4288, h=2848, cx=3600, cy=500):
    img = np.zeros((h, w, 3), np.uint8)
    m = np.zeros((h, w), np.uint8)
    img[cy - 20:cy + 20, cx - 20:cx + 20] = 255          # bright lesion, off-centre horizontally
    m[cy - 20:cy + 20, cx - 20:cx + 20] = 255
    Image.fromarray(img).save(tmp_path / "i.png")
    Image.fromarray(m).save(tmp_path / "m.png")
    return cx, cy


def test_image_and_mask_stay_aligned_on_idrid_sized_photo(tmp_path):
    cx, cy = _scene(tmp_path)
    im, masks, lb = load_pair(tmp_path / "i.png", 256, {"lesion": tmp_path / "m.png"})
    ys, xs = np.nonzero(im[..., 0] > 128)
    ym, xm = np.nonzero(masks["lesion"])
    assert abs(xs.mean() - xm.mean()) < 1.0 and abs(ys.mean() - ym.mean()) < 1.0
    ex, ey = lb.point(cx, cy)
    assert abs(xm.mean() - ex) < 1.5 and abs(ym.mean() - ey) < 1.5


def test_full_field_of_view_is_kept(tmp_path):
    cx, cy = _scene(tmp_path)  # lesion at x=3600 of 4288: the OLD centre-crop (x in 720..3568) dropped it
    im, _, _ = load_pair(tmp_path / "i.png", 256)
    assert (im[..., 0] > 128).sum() > 0


def test_old_pipeline_was_misaligned(tmp_path):
    """Documents the bug: centre-crop for images + whole-mask resize for masks."""
    cx, cy = _scene(tmp_path)
    img = Image.open(tmp_path / "i.png")
    w, h = img.size
    s = min(w, h)
    old = np.asarray(img.crop(((w - s) // 2, 0, (w - s) // 2 + s, s)).resize((256, 256)))
    oldm = np.asarray(Image.open(tmp_path / "m.png").resize((256, 256), Image.NEAREST)) > 127
    xs = np.nonzero(old[..., 0] > 128)[1]
    xm = np.nonzero(oldm)[1]
    assert len(xs) == 0 and len(xm) > 0  # the lesion vanished from the image but stayed in the mask


def test_size_mismatch_raises(tmp_path):
    _scene(tmp_path)
    Image.fromarray(np.zeros((100, 100), np.uint8)).save(tmp_path / "bad.png")
    try:
        load_pair(tmp_path / "i.png", 128, {"lesion": tmp_path / "bad.png"})
        assert False
    except ValueError:
        pass
