"""Deployment cost of the restorer (colleague point 26): size, latency, peak RAM, int8 quantisation, pruning, effect on image quality.
Run the SAME script on the target board (Raspberry Pi etc.) and send results/deployment_<label>.json; numbers measured on another machine
are labelled and only ESTIMATED for the board.   python scripts/10_deployment_benchmark.py --size 128 --label laptop"""
import argparse
import copy
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.utils.prune as prune
from onnxruntime.quantization import QuantType, quantize_dynamic
from skimage.metrics import peak_signal_noise_ratio as psnr

from _common import ROOT
from frugaloptix.ml import Restorer, load_cache, split_index, to_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--rest", default="runs/restorer_selected.pt")
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--label", default=platform.node() or "this-machine")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
net = Restorer(16)
net.load_state_dict(torch.load(a.rest, map_location="cpu"))
net.eval()
x = torch.rand(1, 3, a.size, a.size)


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return dict(median_ms=float(np.median(t)), p90_ms=float(np.percentile(t, 90)))


def onnx_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


Path("runs").mkdir(exist_ok=True)
fp32, int8 = "runs/restorer_bench_fp32.onnx", "runs/restorer_bench_int8.onnx"
torch.onnx.export(net, x, fp32, input_names=["image"], output_names=["restored"], opset_version=17, dynamo=False)
quantize_dynamic(fp32, int8, weight_type=QuantType.QUInt8)
pruned = copy.deepcopy(net)
for m in pruned.modules():
    if isinstance(m, torch.nn.Conv2d) and m.weight.shape[0] > 3:
        prune.l1_unstructured(m, "weight", 0.3)
        prune.remove(m, "weight")
idx = split_index(a.cache)
va = np.array(idx["split"]["val"])[:24]
A, C = load_cache(a.cache, "A")[va], load_cache(a.cache, "C")[va]
sess32, sess8 = onnx_session(fp32), onnx_session(int8)


def quality(fn):
    o = np.concatenate([fn(C[i:i + 4]) for i in range(0, len(C), 4)])
    return float(np.mean([psnr(A[i], o[i], data_range=255) for i in range(len(A))]))


def t_fn(m):
    return lambda c: (m(to_tensor(c)).detach().clamp(0, 1).numpy().transpose(0, 2, 3, 1) * 255 + .5).astype(np.uint8)


def o_fn(s):
    return lambda c: np.concatenate([(np.clip(s.run(None, {"image": to_tensor(c[i:i + 1]).numpy()})[0], 0, 1).transpose(0, 2, 3, 1) * 255 + .5).astype(np.uint8) for i in range(len(c))])


rows = {}
with torch.no_grad():
    rows["torch_fp32"] = dict(**bench(lambda: net(x)), psnr_db=quality(t_fn(net)))
    rows["torch_pruned30"] = dict(**bench(lambda: pruned(x)), psnr_db=quality(t_fn(pruned)), note="unstructured 30% weight pruning: no speed-up without sparse kernels")
xn = x.numpy()
rows["onnx_fp32"] = dict(**bench(lambda: sess32.run(None, {"image": xn})), psnr_db=quality(o_fn(sess32)), size_mb=os.path.getsize(fp32) / 1e6)
rows["onnx_int8_dynamic"] = dict(**bench(lambda: sess8.run(None, {"image": xn})), psnr_db=quality(o_fn(sess8)), size_mb=os.path.getsize(int8) / 1e6)
raw = float(np.mean([psnr(A[i], C[i], data_range=255) for i in range(len(A))]))
for r in rows.values():
    r["psnr_gain_over_raw_db"] = r["psnr_db"] - raw
rows["onnx_fp32"]["peak_rss_mb_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() == "Linux" else 1024 * 1024)
params = int(sum(p.numel() for p in net.parameters()))
macs = None
try:
    macs = float(sum(2 * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] for m in net.modules() if isinstance(m, torch.nn.Conv2d)))  # per output pixel, upper bound
except Exception:
    pass
out = dict(label=a.label, machine=platform.platform(), cpu=platform.processor() or platform.machine(), threads=a.threads, image_px=a.size, params=params, rows=rows,
           raw_polymer_psnr_db=raw, power_w="NOT measured (needs a USB power meter on the board)",
           board_estimate=dict(label="Estimated, not measured", raspberry_pi_zero2w_ms=[float(rows["onnx_int8_dynamic"]["median_ms"] * f) for f in (4, 8)],
                               raspberry_pi4_ms=[float(rows["onnx_int8_dynamic"]["median_ms"] * f) for f in (1.5, 3)], basis="typical single-core slow-down factors vs a desktop-class core; run this script on the board to replace"),
           not_done=["knowledge distillation", "structured pruning", "energy per scan", "full-pipeline (IQA + ensemble + classifier) timing on the board"])
Path(ROOT / "results" / f"deployment_{a.label}.json").write_text(json.dumps(out, indent=1))
(ROOT / "results" / "deployment.json").write_text(json.dumps(out, indent=1))
for k, r in rows.items():
    print(f"{k:<20} {r['median_ms']:.1f} ms (p90 {r['p90_ms']:.1f}) PSNR {r['psnr_db']:.2f} dB (gain {r['psnr_gain_over_raw_db']:+.2f}) {r.get('size_mb', '')}")
