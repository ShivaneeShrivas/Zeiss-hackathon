"""Training loops shared by the scripts (classifier, restorer) with ON-THE-FLY randomised degradation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from frugaloptix.ml import Classifier, Restorer, device, to_tensor
from frugaloptix.render import DegradationSampler, degrade_torch


def _fixed_val_set(A, idx, sampler_seed, nominal, mc, thermal, px_um, dev):
    """Degrade the validation images ONCE with a fixed seed so epochs are comparable. Stored as uint8."""
    s = DegradationSampler(nominal, mc, thermal, seed=sampler_seed)
    torch.manual_seed(sampler_seed)
    out = []
    for i in range(0, len(idx), 32):
        x = to_tensor(A[idx[i:i + 32]])
        sp, ns = s.sample(len(x))
        out.append((degrade_torch(x, sp, px_um, torch.tensor(ns, dtype=torch.float32)) * 255 + 0.5).byte())
    return torch.cat(out)


def train_classifier(src, y, tr, va, out, epochs=8, bs=32, lr=3e-4, seed=0, pretrained=False, degrade=None, log=print, arch="resnet18"):
    """src: cached uint8 images. degrade=None -> train on src as-is (conditions A or B).
    degrade=dict(nominal, mc, thermal, px_um) -> src must be CLEAN originals; images are degraded on the fly (baseline E)."""
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float32)
    m = Classifier(pretrained, arch).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-6)).to(dev)
    sampler = DegradationSampler(degrade["nominal"], degrade.get("mc"), degrade.get("thermal", 0.0), seed=seed) if degrade else None
    val_fixed = _fixed_val_set(src, va, 999, degrade["nominal"], degrade.get("mc"), degrade.get("thermal", 0.0), degrade["px_um"], dev) if degrade else None
    best = -1.0
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = np.sort(perm[i:i + bs])
            x = to_tensor(src[b])
            if sampler:
                sp, ns = sampler.sample(len(b))
                x = degrade_torch(x, sp, degrade["px_um"], torch.tensor(ns, dtype=torch.float32))
            x = x.to(dev)
            if rng.random() < 0.5:
                x = x.flip(3)
            loss = F.binary_cross_entropy_with_logits(m(x), torch.from_numpy(y[b]).to(dev), pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            if val_fixed is None:
                sc = np.concatenate([torch.sigmoid(m(to_tensor(src[va[i:i + 64]]).to(dev))).cpu().numpy() for i in range(0, len(va), 64)])
            else:
                sc = np.concatenate([torch.sigmoid(m(val_fixed[i:i + 64].float().div(255).to(dev))).cpu().numpy() for i in range(0, len(va), 64)])
        auc = roc_auc_score(y[va], sc)
        log(f"  epoch {ep + 1}/{epochs} train_loss {tot / len(tr):.4f} val_AUC {auc:.4f}")
        if auc >= best:
            best = auc
            torch.save(m.state_dict(), out)
    return best


def _grad(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def restorer_loss(out, orig, wmap, grad_weight):
    """Fidelity + lesion-weighted fidelity + edge (gradient) preservation. NO classifier involved."""
    l1 = (torch.abs(out - orig) * wmap).mean()
    if grad_weight:
        ox, oy = _grad(out)
        tx, ty = _grad(orig)
        l1 = l1 + grad_weight * ((ox - tx).abs().mean() + (oy - ty).abs().mean())
    return l1


def train_restorer(A, y, tr, va, clf, out, degrade, epochs=20, bs=16, lr=1e-3, task_weight=0.0, base=16, seed=0, log=print,
                   masks=None, lesion_weight=0.0, grad_weight=0.0, arch="unet", target_arr=None):
    """Degraded -> original. Loss = (lesion-weighted) L1 + gradient loss [+ optional small classifier term].
    Epoch selection uses ONLY the fidelity validation loss (never the classifier), so the classifier cannot steer
    which checkpoint is kept. Findings (docs/RESTORER_STUDY.md): a classifier loss alone made PSNR worse than doing nothing."""
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float32)
    if clf is not None:
        clf.eval().to(dev)
        for p in clf.parameters():
            p.requires_grad_(False)
    net = Restorer(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    sampler = DegradationSampler(degrade["nominal"], degrade.get("mc"), degrade.get("thermal", 0.0), seed=seed)
    val_in = _fixed_val_set(A, va, 999, degrade["nominal"], degrade.get("mc"), degrade.get("thermal", 0.0), degrade["px_um"], dev)
    from scipy.ndimage import binary_dilation

    def wmap_for(ix, flip=False):
        if masks is None or not lesion_weight:
            return torch.ones(len(ix), 1, 1, 1, device=dev)
        m = np.stack([binary_dilation(masks[i], iterations=2) for i in ix]).astype(np.float32)
        w = (1 + lesion_weight * torch.from_numpy(m)[:, None]).to(dev)
        return w.flip(3) if flip else w

    def step(x, o, ix, yb, train, flip=False):
        out_ = net(x)
        loss = restorer_loss(out_, o, wmap_for(ix, flip), grad_weight)
        fid = loss.item()
        if task_weight and clf is not None:
            loss = loss + task_weight * F.binary_cross_entropy_with_logits(clf(out_), yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), fid

    best = 1e9
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        tl = []
        for i in range(0, len(perm), bs):
            b = np.sort(perm[i:i + bs])
            o = to_tensor(A[b])
            sp, ns = sampler.sample(len(b))
            x = degrade_torch(o, sp, degrade["px_um"], torch.tensor(ns, dtype=torch.float32))
            if target_arr is not None:
                o = to_tensor(target_arr[b])        # e.g. the GLASS render: restore to what the reference lens would have delivered
            fl = bool(rng.random() < 0.5)
            if fl:
                x, o = x.flip(3), o.flip(3)
            tl.append(step(x.to(dev), o.to(dev), b, torch.from_numpy(y[b]).to(dev), True, fl)[0])
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = [step(val_in[i:i + bs].float().div(255).to(dev), to_tensor((target_arr if target_arr is not None else A)[va[i:i + bs] if False else va[i:i + bs]]).to(dev), va[i:i + bs], torch.from_numpy(y[va[i:i + bs]]).to(dev), False)[1]
                  for i in range(0, len(va), bs)]
        v = float(np.mean(vl))
        log(f"  epoch {ep + 1}/{epochs} train {np.mean(tl):.4f} val_fidelity_loss {v:.4f}")
        if v < best:
            best = v
            torch.save(net.state_dict(), out)
    return best
