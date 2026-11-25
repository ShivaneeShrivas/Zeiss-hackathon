"""Shared ML helpers: data cache, classifier, restorer (U-Net), batched inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load((ROOT / "configs" / "requirement_sheet.yaml").read_text())
CONDS = ["A", "B", "C", "D"]
COND_NAMES = {"A": "Original image", "B": "Glass lens (4 elements)", "C": "Polymer lens (2 elements), raw",
              "D": "Polymer lens + AI restorer"}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def read_square(path, size: int) -> np.ndarray:
    """Full field of view kept (letterbox, no crop). Use preprocess.load_pair when masks are involved."""
    from frugaloptix.preprocess import load_pair
    return load_pair(path, size)[0]


def cache_dir(cache: str | Path) -> Path:
    d = Path(cache)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cache(cache: str | Path, cond: str) -> np.ndarray:
    return np.load(Path(cache) / f"{cond}.npy", mmap_mode="r")


def split_index(cache: str | Path) -> dict:
    return json.loads((Path(cache) / "index.json").read_text())


def to_tensor(batch_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(batch_u8)).permute(0, 3, 1, 2).float() / 255.0


# ---------------- models ----------------
class Classifier(nn.Module):
    """arch='resnet18' (default, used to supervise the restorer) or 'mobilenet' (an INDEPENDENT evaluator of restored images)."""
    def __init__(self, pretrained: bool = False, arch: str = "resnet18"):
        super().__init__()
        if arch == "mobilenet":
            w = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.net = torchvision.models.mobilenet_v3_small(weights=w)
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, 1)
        else:
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            self.net = torchvision.models.resnet18(weights=w)
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):  # x in [0,1]
        return self.net((x - self.mean) / self.std).squeeze(1)


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.GELU())


class Restorer(nn.Module):
    """Small U-Net predicting a residual correction. ~0.5M params at base=16 (phone-friendly)."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(3, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.d2, self.d1 = _block(base * 6, base * 2), _block(base * 3, base)
        self.out = nn.Conv2d(base, 3, 1)
        nn.init.zeros_(self.out.weight)  # start as identity; learn only the correction
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return (x + self.out(d1)).clamp(0, 1)


@torch.no_grad()
def predict(model: nn.Module, arr_u8: np.ndarray, idx, bs: int = 32, dev=None, fn=None) -> np.ndarray:
    dev = dev or device()
    model.eval().to(dev)
    outs = []
    for i in range(0, len(idx), bs):
        x = to_tensor(arr_u8[idx[i:i + bs]]).to(dev)
        if fn is not None:
            x = fn(x)
        outs.append(x.cpu().numpy() if x.ndim == 4 else x.cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def classify(clf: nn.Module, arr_u8, idx, bs: int = 32, dev=None, pre=None) -> np.ndarray:
    dev = dev or device()
    clf.eval().to(dev)
    out = []
    for i in range(0, len(idx), bs):
        x = to_tensor(arr_u8[idx[i:i + bs]]).to(dev)
        if pre is not None:
            x = pre(x)
        out.append(torch.sigmoid(clf(x)).cpu().numpy())
    return np.concatenate(out)
