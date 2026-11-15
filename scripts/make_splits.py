"""Build fixed, checksummed splits from a labels CSV (e.g. RFMiD: columns ID, Disease_Risk).

python scripts/make_splits.py --labels data/RFMiD/labels.csv --id-col ID --label-col Disease_Risk \
       --images data/RFMiD/images --out data/splits.json
Requires the datasets from IEEE DataPort (account + data-use terms); they are not stored in this repo.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frugaloptix.splits import stratified_split, write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--id-col", default="ID")
ap.add_argument("--label-col", default="Disease_Risk")
ap.add_argument("--images", required=True)
ap.add_argument("--out", default="data/splits.json")
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()
rows = list(csv.DictReader(open(a.labels)))
ids = [r[a.id_col] for r in rows]
y = [int(r[a.label_col]) for r in rows]
splits = stratified_split(ids, y, seed=a.seed)
m = write_manifest(a.images, a.out, splits)
print({k: len(v) for k, v in splits.items()}, "files hashed:", m["n_files"])
