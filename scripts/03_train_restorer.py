"""Restorer trained on RANDOMISED degradations (tolerance Monte Carlo + thermal defocus + variable noise),
with a lesion-aware task loss from the frozen classifier. Never sees the independent OOD renderer."""
import argparse

import numpy as np
import torch

from _common import ROOT, degrade_cfg  # noqa: F401
from frugaloptix.ml import SPEC, Classifier, load_cache, split_index
from frugaloptix.train import train_restorer

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--clf", default="runs/clf_B_s0.pt")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--variant", choices=["fid", "lesion", "lesion_task", "lesion_glass"], default="lesion",
                help="lesion_glass: lesion-preserving, but the TARGET is the glass render (parity with the reference lens) | fid: L1 | lesion: L1 + lesion-weighted L1 + gradient (default) | lesion_task: lesion + small classifier term (ablation)")
ap.add_argument("--lesion-weight", type=float, default=8.0)
ap.add_argument("--grad-weight", type=float, default=1.0)
ap.add_argument("--task-weight", type=float, default=0.01, help="only for --variant lesion_task")
ap.add_argument("--base", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pretrained", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()
idx = split_index(a.cache)
y = np.array(idx["labels"])
tr, va = np.array(idx["split"]["train"]), np.array(idx["split"]["val"])
clf = Classifier(a.pretrained)
clf.load_state_dict(torch.load(a.clf, map_location="cpu"))
out = a.out or f"runs/restorer_{a.variant}_s{a.seed}.pt"
cfg = degrade_cfg(SPEC)
print("training degradations: nominal", [round(v, 1) for v in cfg["nominal"]], "| MC spot sets:", None if cfg["mc"] is None else len(cfg["mc"]), "| thermal rms um:", round(cfg["thermal"], 2))
masks = np.load(f"{a.cache}/masks.npy") if (a.variant != "fid" and __import__("os").path.exists(f"{a.cache}/masks.npy")) else None
best = train_restorer(load_cache(a.cache, "A"), y, tr, va, clf if a.variant == "lesion_task" else None, out, cfg, a.epochs, base=a.base, seed=a.seed,
                      task_weight=a.task_weight if a.variant == "lesion_task" else 0.0, masks=masks,
                      target_arr=load_cache(a.cache, "B") if a.variant == "lesion_glass" else None, lesion_weight=0.0 if a.variant == "fid" else a.lesion_weight, grad_weight=0.0 if a.variant == "fid" else a.grad_weight)
print("saved best ->", out, "val_loss", round(best, 4))
