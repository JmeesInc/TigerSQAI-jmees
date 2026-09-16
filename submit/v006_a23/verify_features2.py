"""コンテナ側 t3_features.py が学習側と同じ値を出すかを実マスクで突き合わせる（在庫 + 解剖文脈 + 粗グリッド）.

学習側:
  features_from_masks.py  (在庫, 1/4 間引き)
  features_anatomy.py     (接触行列 / 周囲クラス / 近傍性, 1/4 間引き)
  features_grid.py        (6x12 セル占有率, gh*8 x gw*8 に縮小)
コンテナ側は argmax ID マップから同じ関数を呼ぶ。LUT は画素独立なので、
「RGB→ID してから縮小」と「縮小してから RGB→ID」は最近傍縮小なら一致する。

Usage: python3 verify_features2.py [--n 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import t3_features as T  # noqa: E402

PRED = REPO / "workspace/expE01_ensemble/results/candE_new9"   # task1=fine 色 / task2=coarse 色
CSV = REPO / "workspace/expT04_task3_sweep"


def build_luts():
    lm = pd.read_csv(REPO / "data/labelmap.csv")
    lf = np.zeros(256 ** 3, np.uint8); lc = np.zeros(256 ** 3, np.uint8)
    for _, r in lm.iterrows():
        lf[int(r.fine_r) * 65536 + int(r.fine_g) * 256 + int(r.fine_b)] = int(r.fine_id)
        lc[int(r.merged_r) * 65536 + int(r.merged_g) * 256 + int(r.merged_b)] = int(r.merged_id)
    return lf, lc


def decode(p, lut, sub=None, size=None):
    b = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if sub:
        b = cv2.resize(b, (b.shape[1] // sub, b.shape[0] // sub), interpolation=cv2.INTER_NEAREST)
    if size:
        b = cv2.resize(b, size, interpolation=cv2.INTER_NEAREST)
    rgb = b[:, :, ::-1].astype(np.uint32)
    return lut[rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
    lf, lc = build_luts()
    ref = {n: pd.read_csv(CSV / f"features_{n}_candE.csv").set_index("case_id")
           for n in ("anat", "grid")}
    ref["inv"] = pd.read_csv(CSV / "features_candE_fix.csv").set_index("case_id")
    files = sorted((PRED / "task1").glob("*.png"))[: a.n]
    worst = 0.0
    for f in files:
        fine = decode(f, lf, sub=T.SUB)
        coarse = decode(PRED / "task2" / f.name, lc, sub=T.SUB)
        got = T.all_features(fine, coarse)
        gh, gw = 6, 12
        cg = decode(PRED / "task2" / f.name, lc, size=(gw * 8, gh * 8))
        got.update(T.grid_features(cg, 16, gh, gw, "c"))
        for kind, df in ref.items():
            row = df.loc[f.stem]
            for k in row.index:
                if k not in got:
                    continue
                d = abs(float(row[k]) - float(got[k]))
                if d > worst:
                    worst, worst_k = d, (f.stem, kind, k, float(row[k]), float(got[k]))
    print(f"最大差 {worst:.3e}")
    if worst > 1e-6:
        print("不一致:", worst_k)
    assert worst < 1e-5, "学習側と一致しない"
    print(f"OK: {len(files)} 枚で在庫 + 解剖文脈 + 粗グリッドが一致")


if __name__ == "__main__":
    main()
