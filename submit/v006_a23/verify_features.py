"""学習側 (expT04/features_from_masks.py) と提出側 (t3_features.py) の特徴が
**1 ビットも違わない**ことを実マスクで確認する。

Usage: python3 verify_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "workspace/expT04_task3_sweep"))

from t3_features import SUB, mask_features as sub_feat          # noqa: E402
from features_from_masks import build_luts, mask_features as tr_feat  # noqa: E402

pred = REPO / "workspace/expE01_ensemble/results/ens5"
lm = pd.read_csv(REPO / "data/labelmap.csv")
lut_f, _ = build_luts(lm)
n_fine = int(lm.fine_id.max()) + 1
files = sorted((pred / "task2").glob("*.png"))[:5]
assert files
for f in files:
    rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)[::SUB, ::SUB]
    idx = (rgb[..., 0].astype(np.int64) * 65536 + rgb[..., 1].astype(np.int64) * 256
           + rgb[..., 2].astype(np.int64))
    lab = lut_f[idx]
    a, b = tr_feat(lab, n_fine, "f"), sub_feat(lab, n_fine, "f")
    assert set(a) == set(b), "特徴名が不一致"
    d = max(abs(a[k] - b[k]) for k in a)
    print(f"{f.name}: {len(a)} 特徴, 最大差 {d:.2e}")
    assert d == 0.0
print("OK: 学習側と提出側の特徴は完全一致")
